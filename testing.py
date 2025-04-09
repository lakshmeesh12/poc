# testing.py
import os
import json
import glob
import shutil
import docker
import pytest
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from flask import jsonify
import logging
import re

# Setup logging with both file and console handlers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='testing.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration
BASE_TEST_DIR = "Test_Results"
RESULTS_DIR = "Results"
SUPPORTED_LANGUAGES = {
    "py": "Python",
    "js": "Javascript",
    "java": "Java",
    "cs": "Csharp",
    "cpp": "C++",
}

# Ensure base directory exists
os.makedirs(BASE_TEST_DIR, exist_ok=True)

def get_docker_client():
    """Initialize Docker client with timeout and error handling"""
    try:
        client = docker.from_env(timeout=10)
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Docker client: {str(e)}")
        raise RuntimeError("Docker service unavailable. Please ensure Docker Desktop is running.")

def get_language_from_extension(file_extension: str) -> str:
    """Map file extension to language name"""
    return SUPPORTED_LANGUAGES.get(file_extension, "Python")

def enforce_source_import(test_code: str, work_item_id: str) -> str:
    """Post-process test code to enforce 'from source import ...'"""
    logger.debug(f"Enforcing 'from source import ...' for {work_item_id}")
    lines = test_code.split('\n')
    corrected_lines = []
    
    for line in lines:
        if 'from' in line and 'import' in line and 'source' not in line:
            parts = line.split('import')
            import_items = parts[1].strip()
            corrected_line = f"from source import {import_items}"
            logger.debug(f"Corrected import: '{line}' -> '{corrected_line}'")
            corrected_lines.append(corrected_line)
        else:
            corrected_lines.append(line)
    
    corrected_code = '\n'.join(corrected_lines).strip()
    if not any(line.strip().startswith('def test_') for line in corrected_code.split('\n')):
        logger.error(f"No test functions found after correction for {work_item_id}: {corrected_code[:500]}...")
        raise ValueError("Corrected code contains no valid pytest test functions")
    
    return corrected_code

def generate_test_cases(code_content: str, language: str, work_item_id: str) -> str:
    """Generate test cases using OpenAI with strict import enforcement"""
    logger.info(f"Generating test cases for work_item_id: {work_item_id}")
    try:
        prompt = f"""Given the following {language} code from a file renamed to 'source.py', generate comprehensive test cases:
        1. Unit tests
        2. API tests (if applicable)
        3. Functional tests
        4. UI tests (if applicable)
        
        Return ONLY the pytest code in a ```python block. Use EXCLUSIVELY 'from source import ...' for all imports from the original code.
        Include necessary imports, clear test function names, and setup/teardown if needed. NO TEXT OUTSIDE THE ```python BLOCK:
        
        ```{language}
        {code_content}
        ```"""

        logger.debug("Sending initial request to OpenAI")
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a testing expert generating pytest test cases"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        raw_test_code = response.choices[0].message.content.strip()
        pattern = r'```(?:python)?\n([\s\S]*?)\n```'
        match = re.search(pattern, raw_test_code)
        test_code = match.group(1).strip() if match else raw_test_code.strip()

        if 'from source import' not in test_code:
            logger.warning(f"OpenAI ignored import instruction for {work_item_id}, correcting...")
            fix_prompt = f"""The following test code uses incorrect imports. Rewrite it to use ONLY 'from source import ...' for all imports from 'source.py'. Return ONLY the corrected pytest code in a ```python block, NO TEXT OUTSIDE:

            ```python
            {test_code}
            ```"""
            fix_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a code correction expert"},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            fixed_raw_code = fix_response.choices[0].message.content.strip()
            match = re.search(pattern, fixed_raw_code)
            test_code = match.group(1).strip() if match else fixed_raw_code.strip()

        test_code = enforce_source_import(test_code, work_item_id)

        test_dir = f"{BASE_TEST_DIR}/{work_item_id}"
        os.makedirs(test_dir, exist_ok=True)

        test_file = f"{test_dir}/test_generated.py"
        with open(test_file, "w") as f:
            f.write(test_code)

        test_count = len([line for line in test_code.split('\n') if line.strip().startswith('def test_')])
        logger.info(f"Generated {test_count} test cases for {work_item_id}")

        return test_code

    except Exception as e:
        logger.error(f"Failed to generate test cases for {work_item_id}: {str(e)}", exc_info=True)
        raise

def setup_docker_environment(work_item_id: str, language: str) -> Any:
    """Setup Docker container with required dependencies"""
    logger.info(f"Setting up Docker environment for {work_item_id}")
    try:
        docker_client = get_docker_client()
        test_dir = f"{BASE_TEST_DIR}/{work_item_id}"

        dockerfile_content = """
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir pytest pytest-playwright requests pytest-json-report
RUN playwright install --with-deps
COPY . /app
COPY pytest_runner.py /app/pytest_runner.py
CMD ["python", "pytest_runner.py"]
"""

        runner_content = """
import pytest
import json

def run_tests():
    result = pytest.main(["test_generated.py", "-v", "--json-report", "--json-report-file=report.json"])
    return result

if __name__ == "__main__":
    exit(run_tests())
"""

        with open(f"{test_dir}/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        with open(f"{test_dir}/pytest_runner.py", "w") as f:
            f.write(runner_content)

        source_file = glob.glob(f"{RESULTS_DIR}/{work_item_id}.*")[0]
        target_file = f"{test_dir}/source.py"
        shutil.copy(source_file, target_file)
        logger.debug(f"Copied and renamed source file: {source_file} to {target_file}")

        logger.debug(f"Building Docker image for {work_item_id}")
        image, build_logs = docker_client.images.build(
            path=test_dir,
            tag=f"test_{work_item_id}",
            rm=True,
            quiet=False
        )
        
        for log in build_logs:
            if 'stream' in log:
                logger.debug(f"Docker build log: {log['stream'].strip()}")

        return image

    except Exception as e:
        logger.error(f"Failed to setup Docker environment for {work_item_id}: {str(e)}", exc_info=True)
        raise

def execute_tests(work_item_id: str) -> Dict[str, Any]:
    """Execute all tests in Docker container"""
    logger.info(f"Executing tests for {work_item_id}")
    try:
        docker_client = get_docker_client()
        test_dir = f"{BASE_TEST_DIR}/{work_item_id}"

        container = docker_client.containers.run(
            image=f"test_{work_item_id}",
            volumes={os.path.abspath(test_dir): {'bind': '/app', 'mode': 'rw'}},
            detach=True
        )

        logger.debug(f"Container started: {container.id}")
        result = container.wait()
        logs = container.logs().decode('utf-8')
        logger.debug(f"Test execution logs: {logs}")
        
        container.remove()
        logger.debug(f"Container removed: {container.id}")

        report_path = f"{test_dir}/report.json"
        report = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            logger.debug(f"Test report generated: {json.dumps(report, indent=2)[:500]}...")
        else:
            logger.warning(f"No report found at {report_path}")

        return {
            "status": "success" if result["StatusCode"] == 0 else "failed",
            "logs": logs,
            "report": report
        }

    except Exception as e:
        logger.error(f"Failed to execute tests for {work_item_id}: {str(e)}", exc_info=True)
        raise

def generate_human_readable_report(work_item_id: str, test_results: Dict[str, Any]) -> str:
    """Generate a human-readable report using OpenAI from report.json"""
    logger.info(f"Generating human-readable report for {work_item_id}")
    try:
        report_json = test_results["report"]
        prompt = f"""Generate a detailed, human-readable test report in Markdown format based on the following test results:
        - Test results (JSON): ```json\n{json.dumps(report_json, indent=2)}\n```

        The report should:
        - Be technically detailed yet written in natural, easy-to-understand language.
        - Include a **Summary** section with total tests, passed, failed, and applicable testing types (e.g., unit, functional, API, UI) inferred from the results.
        - Include a **Test Case Details** section listing each test's name, inferred purpose, what it likely tests (based on name and outcome), and result (pass/fail with explanation, including failure reason if applicable).
        - Be understandable by anyone, technical or not, explaining what the tests cover and their importance.
        - Use numbered lists for clarity in the Test Case Details section.

        Return ONLY the Markdown content, no extra text or code blocks around it."""

        logger.debug(f"Sending human-readable report request to OpenAI for {work_item_id}")
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at writing clear, technical documentation"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )

        human_readable_report = response.choices[0].message.content.strip()
        logger.debug(f"Generated human-readable report for {work_item_id}: {human_readable_report[:500]}...")

        test_dir = f"{BASE_TEST_DIR}/{work_item_id}"
        report_path = f"{test_dir}/human_readable_report.md"
        os.makedirs(test_dir, exist_ok=True)  # Ensure directory exists
        with open(report_path, "w") as f:
            f.write(human_readable_report)
        logger.info(f"Human-readable report saved to {report_path}")

        return human_readable_report

    except Exception as e:
        logger.error(f"Failed to generate human-readable report for {work_item_id}: {str(e)}", exc_info=True)
        return f"Error generating report: {str(e)}"  # Return error for debugging

def generate_test_report(test_results: Dict[str, Any], work_item_id: str, source_code: str, test_code: str) -> Dict[str, Any]:
    """Generate both JSON and human-readable reports"""
    logger.info(f"Generating test report for {work_item_id}")
    try:
        report_dir = f"{BASE_TEST_DIR}/{work_item_id}"
        report = test_results["report"]
        
        total_tests = report.get("summary", {}).get("total", 0)
        passed = report.get("summary", {}).get("passed", 0)
        failed = report.get("summary", {}).get("failed", 0)
        
        failures = []
        for test in report.get("tests", []):
            if test["outcome"] == "failed":
                failures.append({
                    "test_name": test["nodeid"],
                    "reason": test.get("call", {}).get("crash", {}).get("message", "Unknown failure reason")
                })

        summary = {
            "timestamp": datetime.now().isoformat(),
            "work_item_id": work_item_id,
            "status": test_results["status"],
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "failures": failures,
            "logs": test_results["logs"]
        }

        os.makedirs(report_dir, exist_ok=True)
        json_report_path = f"{report_dir}/report.json"  # Already saved by pytest
        summary_path = f"{report_dir}/summary_report.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        # Generate human-readable report using only report.json
        human_readable_report = generate_human_readable_report(work_item_id, test_results)
        summary["human_readable_report"] = human_readable_report
        
        logger.info(f"Test report summary for {work_item_id}: {json.dumps(summary, indent=2)[:500]}...")
        return summary

    except Exception as e:
        logger.error(f"Failed to generate test report for {work_item_id}: {str(e)}", exc_info=True)
        raise

def run_tests_handler(work_item_id: str):
    """Handler for running tests"""
    logger.info(f"Starting test run for work_item_id: {work_item_id}")
    try:
        code_files = glob.glob(f"{RESULTS_DIR}/{work_item_id}.*")
        if not code_files:
            logger.error(f"No code file found for {work_item_id}")
            return jsonify({"error": f"No code file found for {work_item_id}"}), 404
        
        code_file = code_files[0]
        file_extension = code_file.split('.')[-1]
        language = get_language_from_extension(file_extension)
        logger.debug(f"Found code file: {code_file}, Language: {language}")

        with open(code_file, 'r') as f:
            code_content = f.read()
        logger.debug(f"Code content: {code_content[:500]}...")

        test_code = generate_test_cases(code_content, language, work_item_id)
        setup_docker_environment(work_item_id, language)
        test_results = execute_tests(work_item_id)
        report = generate_test_report(test_results, work_item_id, code_content, test_code)

        return jsonify({
            "message": "Tests executed successfully",
            "report_url": f"/view_test_results/{work_item_id}",
            "summary": report
        })

    except Exception as e:
        logger.error(f"Test run failed for {work_item_id}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def view_test_results_handler(work_item_id: str):
    """Handler for viewing test results"""
    logger.info(f"Viewing test results for {work_item_id}")
    try:
        report_path = f"{BASE_TEST_DIR}/{work_item_id}/summary_report.json"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            human_path = f"{BASE_TEST_DIR}/{work_item_id}/human_readable_report.md"
            if os.path.exists(human_path):
                with open(human_path, 'r') as f:
                    report["human_readable_report"] = f.read()
            else:
                logger.warning(f"Human-readable report not found at {human_path}")
            logger.debug(f"Returning report: {json.dumps(report, indent=2)[:500]}...")
            return jsonify(report)
        else:
            logger.warning(f"Report not found at {report_path}")
            return jsonify({"error": "Report not found"}), 404

    except Exception as e:
        logger.error(f"Failed to view test results for {work_item_id}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def cleanup_test_artifacts(work_item_id: str):
    """Clean up test artifacts"""
    logger.info(f"Cleaning up artifacts for {work_item_id}")
    try:
        test_dir = f"{BASE_TEST_DIR}/{work_item_id}"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            logger.debug(f"Removed test directory: {test_dir}")
        
        docker_client = get_docker_client()
        try:
            docker_client.images.remove(f"test_{work_item_id}", force=True)
            logger.debug(f"Removed Docker image: test_{work_item_id}")
        except docker.errors.ImageNotFound:
            logger.debug(f"No Docker image found to remove: test_{work_item_id}")
            
    except Exception as e:
        logger.error(f"Cleanup failed for {work_item_id}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_work_item_id = "test123"
    os.makedirs(f"{RESULTS_DIR}", exist_ok=True)
    with open(f"{RESULTS_DIR}/{test_work_item_id}.py", "w") as f:
        f.write("def add(a, b):\n    return a + b")
    result = run_tests_handler(test_work_item_id)
    print(result.get_json())