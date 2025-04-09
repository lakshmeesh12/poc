import os
from flask import (
    Flask,
    render_template,
    send_from_directory,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash,
)
from flask_cors import CORS
import hashlib
import sqlite3
from dotenv import load_dotenv
from jira_api import JIRAWrapper
from flask import Flask, render_template, jsonify
from ado_api import ADOWrapper

from openai_api import response_from_llm
from codellama_api import codellama_response
from anthropic_api import anthropic_response_from_llm
import glob, json
import glob
import json
from testing import run_tests_handler, view_test_results_handler
from test_results import view_results_handler
import logging



# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)
app.secret_key = "10"
open_api_key = os.environ.get("OPENAI_API_KEY")
print(open_api_key)
ADO_organization = "QuadrantPOC"
ADO_project_name = "Test_Project_ADOIntegrationPOC"
ADO_repository_name = "JIRA_Integration"
JIRA_project = "TestProjectJiraintegration"

Languages_supported = {
    "VBA":"vba",
    "Python": "py",
    "Javascript": "js",
    "Java": "java",
    "Csharp": "cs",
    "Html": "html",
    "CSS": "css",
    "C": "c",
    "C++": "cpp",
    "Scala": "scala",
    "SQL": "sql",
}
db_name = "sql.db"
jira_wrapper = JIRAWrapper(
    os.environ.get("JIRA_API_TOKEN"),
    project=JIRA_project,
    useremail="venkata.sai@quadranttechnologies.com",
)

ado_wrapper = ADOWrapper(
    os.environ.get("AZURE_DEVOPS_PAT"),
    organization=ADO_organization,
    project=ADO_project_name,
    repository_name=ADO_repository_name,
)
print(os.environ.get("AZURE_DEVOPS_PAT"))

def create_directory_if_not_exists(directory):
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)


def admin_portal():
    pass


def get_user_data_from_db(email, password):
    # Hash the provided password for comparison
    hashed_password = hashlib.sha256(password.encode("utf-8")).hexdigest()

    # Check if the user exists in the 'users' table
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name,email FROM users WHERE email = ? AND password = ? ",
        (email, hashed_password),
    )
    user = cursor.fetchone()
    return user


@app.route("/login", methods=["GET", "POST"])
def login():
    print("Login is triggered")
    if request.method == "POST":
        print("POST")
        form_data = request.form.to_dict()
        email = form_data["email"]
        password = form_data["password"]
        user_data = get_user_data_from_db(email=email, password=password)
        print(user_data)
        if user_data:
            session["username"] = user_data[0]
            session["user_email"] = user_data[1]
            # os.environ["AZURE_DEVOPS_PAT"] = str(user_data[4])
            print("Redirecting to home")
            return redirect(url_for("home"))
        else:
            print("Redirecting to Signup")
            return redirect(url_for("signup"))

        # print(render_template("login.html"))
    print("Rendering Login")
    return render_template("login.html")


def check_db_for_user(email):
    SQL_Query = f"""SELECT * from users where email='{email}'"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(SQL_Query)
    result = cursor.fetchone()

    return result


@app.route("/logout")
def logout():
    session.pop("username", None)  # Remove the user's session data
    return redirect(
        url_for("home")
    )  # Redirect to the login page or another appropriate route


def insert_new_user_into_db(name, email, hashed_password, pat, manager_email, skills):
    try:

        conn = sqlite3.connect("sql.db")
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (name, email, password, pat,manager_email,skills) VALUES (?, ?, ?, ?,?,?)",
            (name, email, hashed_password, pat, manager_email, skills),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(e)
        return False


@app.route("/signup", methods=["GET", "POST"])
def signup():
    print("Signup is triggered")
    if request.method == "POST":
        print("User Signing up")
        form_data = request.form.to_dict()
        hashed_password = hashlib.sha256(
            form_data["password"].encode("utf-8")
        ).hexdigest()
        if check_db_for_user(form_data["email"]) is None:

            response_from_db = insert_new_user_into_db(
                name=form_data["name"],
                email=form_data["email"],
                hashed_password=hashed_password,
                pat=form_data["PAT"],
                manager_email=form_data["manager_email"],
                skills=form_data["skills"],
            )
            print(response_from_db)
            if response_from_db:
                session["username"] = form_data["name"]
                return redirect(url_for("home"))
            else:
                flash("Something went wrong!!", "exists")
                return render_template("signup.html")
        else:
            # flash("user already present in the DB, Please Login!!","exists")
            print("User already in DB")
            print(url_for("login"))
            return redirect(url_for("login"))
    print("GET so rendering Signup !!!")
    return render_template("signup.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    """
    Serve static files from the 'static' directory.

    This route handles requests for static files (e.g., CSS, JavaScript, images)
    by serving them from the 'static' directory. The 'filename' parameter in
    the URL specifies the path to the requested static file within the 'static'
    directory.

    Args:
        filename (str): The path to the requested static file.

    Returns:
        Flask response: The requested static file as a response.
    """
    return send_from_directory("static", filename)


def logged_in():
    return "username" in session


def get_file_name_from_desc(desc: str):
    import re

    # Define a regular expression pattern to extract the filename
    pattern = r"Use_File=([\w\.]+);"

    # Use re.search() to find the pattern in the string
    match = re.search(pattern, desc)
    if match:
        return match.group(1)
    else:
        None


def get_output_file_name_from_desc(desc: str):
    import re

    # Define a regular expression pattern to extract the filename
    pattern = r"Output_Filename=([\w\.]+);"

    # Use re.search() to find the pattern in the string
    match = re.search(pattern, desc)
    if match:
        return match.group(1)
    else:
        None


def get_authenticated_users_list(manager_email):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT email FROM users where manager_email = ? ", (manager_email,))
    user = cursor.fetchall()
    return user


@app.route("/send_option", methods=["POST"])
def send_option():
    data = request.get_json()
    selected_option = data["option"]
    # session["model"] = "gpt-3.5"
    session["model"] = selected_option
    print(
        f"Selected option: {selected_option}"
    )  # You can replace print with any further processing you want
    return jsonify({"message": "Model Updated successfully"})


def get_skills_from_user(email):
    try:

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT skills FROM users where email = ? ", (email,))
        user = cursor.fetchall()
        return user
        # return True
    except Exception as e:
        print(e)
        return False


@app.route("/", methods=["GET", "POST"])
def home():
    print(logged_in())
    if logged_in():
        username = session.get("username")
        user_email = session.get("user_email")
        print(user_email)
        users_list = get_authenticated_users_list(user_email)
        session["skill"] = get_skills_from_user(user_email)[0][0]

        # print(users_list)
        # print(username)
    else:
        username = "Guest"
        user_email = "guest@example.com"
        users_list = []
        print("Not")
        session["skill"] = "None"
    if len(users_list) > 0 and type(users_list[0]) == tuple:
        users_list = [item[0] for item in users_list]
    # print(users_list)
    users_list.append(user_email)

    print(session["skill"])
    print(users_list)
    final_works = []
    if users_list != ["guest@example.com"]:
        response = jira_wrapper.get_issue_details(users=users_list)
        # print(response)

        for work_item in response["issues"]:
            temp = {}

            temp["id"] = work_item["id"]
            temp["title"] = work_item["fields"]["summary"]
            temp["state"] = work_item["fields"]["status"]["name"]
            temp["description"] = work_item["fields"]["description"]
            temp["Assignedto"] = work_item["fields"]["assignee"]["displayName"]
            attachments = jira_wrapper.get_attachment_name_url(i_id=temp["id"])
            temp["attachments"] = []
            # print(attachments)
            if type(attachments) == dict and attachments != {}:

                # if attachments != {} and type(attachments) == dict:

                for attachment_name, attachment_id in attachments.items():
                    temp["attachments"].append(
                        {
                            "name": attachment_name,
                            "url": f"https://quadcode.atlassian.net/rest/api/2/attachment/content/{attachment_id}",
                        }
                    )
            else:
                temp["attachments"].append(
                    {"name": "No attachments Found!!", "url": "#"}
                )
            temp["option1_dropdown"] = list(Languages_supported.keys())
            temp["option2_dropdown"] = list(Languages_supported.keys())
            # print("#" * 25)
            final_works.append(temp)

    else:
        pass
        # print(response)

    """wrk_item_urls = [resp["url"] for resp in response["issues"]]
    work_items = [
        jira_wrapper.get_workitem_details_from_url(url=url)
        for url in wrk_item_urls
    ]"""
    # print(response)

    # print(final_works)

    return render_template(
        "index.html", work_items=final_works, username=username, logged_in=logged_in()
    )


def coalesce(*args):
    return next((arg for arg in args if arg not in (None, "None")), None)


def get_prompt(skill):
    with open(f"prompts/{skill}.txt".lower(), "r") as sys_prompt_file:
        system_prompt = sys_prompt_file.read()
        return system_prompt


def create_file_for_push(
    skill, work_item_id, output_file_name_exp, Languages_supported, code, ado_wrapper
):
    code = code.replace(f"```{skill.lower()}", "")
    code = code.replace(f"```{Languages_supported[skill].lower()}", "")
    code = code.replace(f"```{Languages_supported[skill]}", "")
    code = code.replace("```", "")
    print(f"Skill is {skill}")
    file_name = f"{output_file_name_exp}.{Languages_supported[skill]}"
    print(file_name)
    with open(f"Results/{work_item_id}.txt", "w") as file:
        file.write(code)
    with open(f"Results/{work_item_id}.{Languages_supported[skill]}", "w") as file:
        file.write(code)

    response = ado_wrapper.create_repo_and_push(
        branch_name=f"branch_{work_item_id}",
        file_name=file_name,
        code=code,
    )
    print("Response status from ADO:", response.status_code)
    if response.status_code == 201:
        return file_name
    else:
        return "None"


def get_vba_from_excel():
    pass


@app.route("/generate_code/<work_item_id>/", methods=["POST"])
def generate_code(work_item_id):
    if request.method == "POST":
        print("POST!!!!!!!!!!!")
        data = request.get_json()
        dropdown1 = data.get("dropdown1")
        dropdown2 = data.get("dropdown2")
        print("dropdown options:")
        print(dropdown1, dropdown2)
        customprompt = data.get("customprompt")
        print(customprompt)
        print(work_item_id)
        if str(customprompt) == "(Optional)":
            customprompt = ""
        work_item = jira_wrapper.get_issue_details_from_id(work_item_id)
        attachmentscontent = jira_wrapper.get_attachment_content(work_item_id)
        file_name = get_file_name_from_desc(str(work_item["fields"]["description"]))
        output_file_name = get_output_file_name_from_desc(
            desc=str(work_item["fields"]["description"])
        )
        if output_file_name is not None:
            session["output_file_name"] = output_file_name.split(".")[0]
            output_file_name_exp = output_file_name.split(".")[0]
        else:
            session["output_file_name"] = work_item_id
            output_file_name_exp = work_item_id
        print(file_name, output_file_name)
        if file_name is not None:
            attachmentscontent += ado_wrapper.get_file_content_from_git(file_path=file_name)
            print(f"Got the file from {file_name}")
        skill_required = coalesce(dropdown2, dropdown1)
        if skill_required == "None":
            skill_required = "old_prompt"
            session["skill"] = "General"
        else:
            session["skill"] = skill_required
        print("Skill Req Final:", skill_required)
        system_prompt = get_prompt(skill=skill_required)
        if dropdown2 != "None" and dropdown1 != "None":
            print("Migration task")
            system_prompt += f"""/n This is a Migration task so, look at the attachments carefully and migrate it from {dropdown1} to {dropdown2}"""
        inputs_to_LLM = {
            "system_prompt": system_prompt,
            "Title": str(work_item["fields"]["summary"]),
            "Description": str(work_item["fields"]["description"]),
            "attachmentscontent": str(attachmentscontent),
            "user_input": customprompt,
        }
        create_directory_if_not_exists(f"Logs/{work_item_id}")
        with open(f"Logs/{work_item_id}/inputs_to_LLM.json", "w") as file:
            json.dump(inputs_to_LLM, file, indent=4)
        print("Model in selection:", session["model"])
        if session["model"] == "gpt-3.5" or session["model"] == "gpt-4":
            code = response_from_llm(model=session["model"], api_key=open_api_key, **inputs_to_LLM)
        elif session["model"] == "codellama":
            response = codellama_response(**inputs_to_LLM)
            print(response)
            print(type(response))
        elif session["model"].startswith("claude"):
            code = anthropic_response_from_llm(
                model=session["model"],
                api_key=os.environ.get("ANTHROPIC_CLAUDE_API_KEY"),
                **inputs_to_LLM,
            )
            code = code[0].text
        with open(f"Logs/{work_item_id}/output.{Languages_supported[skill_required]}", "w") as file:
            file.write(code)
        if len(ADO_project_name.split(" ")) > 1:
            project_url = "%20".join(list(ADO_project_name.split(" ")))
        else:
            project_url = ADO_project_name
        output_file_name_exp = create_file_for_push(
            skill=skill_required,
            work_item_id=work_item_id,
            output_file_name_exp=output_file_name_exp,
            Languages_supported=Languages_supported,
            code=code,
            ado_wrapper=ado_wrapper,
        )
        print("Getting url")
        url = f"https://dev.azure.com/{ADO_organization}/{project_url}/_git/{ADO_repository_name}?path=/ai_generated_{output_file_name_exp}&version=GBbranch_{work_item_id}"
        print(url)
        print("CODE READY")
        print("@" * 25)
        if output_file_name_exp != "None":
            print("Git updated!!!")
            return jsonify({
                "generated_code_link": url,
                "show_code_url": f"/response_page/{work_item_id}/",
                "test_url": f"/run_tests/{work_item_id}/"  # Added test option
            })
        else:
            print("Something went wrong!!")
            return redirect(url_for("home"))


@app.route("/submit_code/<work_item_id>/", methods=["POST"])
def submit_code(work_item_id):
    try:
        print("Getting code from front end")
        file_name = [i for i in glob.glob(f"Results/{work_item_id}.*") if i.split(".")[-1] != "txt"][0]
        code = request.json["code"]
        session["skill"] = list(Languages_supported.keys())[list(Languages_supported.values()).index(file_name.split(".")[-1])]
        if len(ADO_project_name.split(" ")) > 1:
            project_url = "%20".join(list(ADO_project_name.split(" ")))
        else:
            project_url = ADO_project_name
        work_item = jira_wrapper.get_issue_details_from_id(work_item_id)
        output_file_name = get_output_file_name_from_desc(desc=str(work_item["fields"]["description"]))
        if output_file_name is not None:
            session["output_file_name"] = output_file_name.split(".")[0]
            output_file_name_exp = output_file_name.split(".")[0]
        else:
            session["output_file_name"] = work_item_id
            output_file_name_exp = work_item_id
        skill_required = session["skill"]
        print("#" * 25)
        print(output_file_name_exp)
        output_file_name_exp = create_file_for_push(
            skill=skill_required,
            work_item_id=work_item_id,
            output_file_name_exp=output_file_name_exp,
            Languages_supported=Languages_supported,
            code=code,
            ado_wrapper=ado_wrapper,
        )
        print("Getting url")
        url = f"https://dev.azure.com/{ADO_organization}/{project_url}/_git/{ADO_repository_name}?path=/ai_generated_{output_file_name_exp}&version=GBbranch_{work_item_id}"
        print(url)
        print("CODE READY")
        print("@" * 25)
        if output_file_name_exp != "None":
            print("Git updated!!!")
            return jsonify({
                "generated_code_link": url,
                "test_url": f"/run_tests/{work_item_id}/"  # Added test option
            })
        else:
            return jsonify({
                "genated_code_link": f"Unable to show link, Please proceed to Branch: branch_{work_item_id}"
            })
    except Exception as e:
        print("Error in submitting code")
        return jsonify({"error": str(e), "message": "Error processing code"}), 500


@app.route("/run_tests/<work_item_id>/", methods=["GET"])
def run_tests(work_item_id):
    return run_tests_handler(work_item_id)

@app.route("/view_results/<work_item_id>/", methods=["GET"])
def view_results(work_item_id):
    return view_results_handler(work_item_id)


@app.route("/response_page/<work_item_id>/", methods=["GET"])
def show_response(work_item_id):
    # Extract the generated code from the query parameters
    file_name = [
        i for i in glob.glob(f"Results/{work_item_id}.*") if i.split(".")[-1] != "txt"
    ][0]
    if request.method == "GET":
        with open(file_name, "r") as file:
            generated_code = file.read()
        # if session["skill"] == "Python":
        #     with open(f"Results/{work_item_id}.py", "r") as file:
        #         generated_code = file.read()
        # elif session["skill"] == "Javascript":
        #     with open(f"Results/{work_item_id}.js", "r") as file:
        #         generated_code = file.read()
        # generated_code = request.args.get("code")

        # Render the response page with the generated code
        return render_template(
            "code.html", code=str(generated_code), work_item_id=str(work_item_id)
        )


def get_dynamic_files_to_exclude():

    # Specify the pattern to match dynamically created files
    dynamic_files_pattern = "Results/*.py"

    # Use glob to match files based on the pattern
    excluded_files = glob.glob(dynamic_files_pattern)

    return excluded_files


if __name__ == "__main__":
    # excluded_files = get_dynamic_files_to_exclude()
    app.run(port=5000,host='0.0.0.0')  # , use_reloader=False, extra_files=excluded_files)