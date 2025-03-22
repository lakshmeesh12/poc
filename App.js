import React, { useState } from "react";
import { extractInvoices, listBuckets, listFilesInBucket, extractInvoicesS3 } from "./api";
import "./home.css";

const App = () => {
  const [view, setView] = useState("storage_selection");
  const [buckets, setBuckets] = useState([]);
  const [selectedBucket, setSelectedBucket] = useState(null);
  const [filesInBucket, setFilesInBucket] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [files, setFiles] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [error, setError] = useState(null);

  const handleStorageSelect = async (storage) => {
    setError(null);
    if (storage === "local") {
      setView("local_upload");
    } else if (storage === "aws") {
      setLoadingMessage("Fetching Buckets");
      setLoading(true);
      try {
        const bucketsData = await listBuckets();
        setBuckets(bucketsData);
        setView("aws_buckets");
      } catch (err) {
        setError("Failed to fetch buckets: " + err.message);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleBucketSelect = async (bucket) => {
    setSelectedBucket(bucket);
    setLoadingMessage("Retrieving Files");
    setLoading(true);
    try {
      const filesData = await listFilesInBucket(bucket);
      setFilesInBucket(filesData);
      setSelectedFiles([]);
      setView("aws_files");
    } catch (err) {
      setError("Failed to fetch files: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileToggle = (file) => {
    setSelectedFiles((prev) =>
      prev.includes(file) ? prev.filter((f) => f !== file) : [...prev, file]
    );
  };

  const handleFileChange = (e) => {
    setFiles(e.target.files);
    setError(null);
  };

  const handleAnalyzeFraud = async () => {
    if (!files || files.length === 0) {
      setError("Please select at least one file to analyze.");
      return;
    }
    setLoadingMessage("Analyzing");
    setLoading(true);
    setError(null);
    try {
      const data = await extractInvoices(files);
      setResults(data);
      setView("results");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeFraudS3 = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError("Please select at least one file to analyze.");
      return;
    }
    setLoadingMessage("Analyzing");
    setLoading(true);
    setError(null);
    try {
      const data = await extractInvoicesS3(selectedBucket, selectedFiles);
      setResults(data);
      setView("results");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStartOver = () => {
    setView("storage_selection");
    setBuckets([]);
    setSelectedBucket(null);
    setFilesInBucket([]);
    setSelectedFiles([]);
    setFiles(null);
    setResults(null);
    setError(null);
  };




  // Create a separate component for the fraud detection UI
  const renderFraudDetection = (invoice) => {
    const { data, csv_validation, annotated_images } = invoice;
    const fields = [
      "What is the chassis number?",
      "What is the engine number?",
      "What is the make?",
      "What is the model?",
      "What is the customer name?",
      "What is the InsuranceNo or policy number?",
      "What is the Vehicle_Insurance_Company or policy provider company?"
    ];
    const isCsvValidated = csv_validation.status === "Validated";
    const mismatches = fields.filter((field) => !data[field]?.csv_validated);
  
    return (
      <div className="fraud-section">
        <h3>Fraud Detection Results</h3>
        {isCsvValidated ? (
          <>
            <p>Matched using primary key: {csv_validation.primary_key}</p>
            <table className="fraud-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Invoice Data</th>
                  <th>Database Data</th>
                </tr>
              </thead>
              <tbody>
                {fields.map((field) => {
                  const fieldName = field.replace("What is the ", "").replace("?", "");
                  const invoiceValue = data[field]?.value || "Not Found";
                  const csvValue = data[field]?.csv_value || "Not Found";
                  const isMismatch = !data[field]?.csv_validated;
                  return (
                    <tr key={field} className={isMismatch ? "mismatch" : ""}>
                      <td>{fieldName}</td>
                      <td>{invoiceValue}</td>
                      <td>{csvValue}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className={`summary ${mismatches.length > 0 ? "error" : "success"}`}>
              {mismatches.length > 0 ? (
                <>
                  <strong>Fraud Detected:</strong> Mismatching values -{" "}
                  {mismatches
                    .map((field) => field.replace("What is the ", "").replace("?", ""))
                    .join(", ")}
                </>
              ) : (
                <>
                  <strong>No Fraud Detected:</strong> All values match with the database.
                </>
              )}
            </div>
            {annotated_images.map((img, idx) => (
              <div key={idx} className="annotated-image-container">
                <img
                  src={`http://localhost:8000${img}`} // Add your backend URL here
                  alt={`Annotated page ${idx + 1}`}
                  className="annotated-image"
                />
                <span className="red-box-label">
                  {invoice.file_type === "PDF" ? `Page ${idx + 1}` : "Image"}
                </span>
              </div>
            ))}
          </>
        ) : (
          <>
            <table className="fraud-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Invoice Data</th>
                </tr>
              </thead>
              <tbody>
                {fields.map((field) => (
                  <tr key={field}>
                    <td>{field.replace("What is the ", "").replace("?", "")}</td>
                    <td>{data[field]?.value || "Not Found"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="summary warning">No matching record found in the database.</div>
            {annotated_images.length > 0 && (
              <div className="annotated-images">
                <h4>Extracted Values in Document</h4>
                {annotated_images.map((img, idx) => (
                  <div key={idx} className="annotated-image-container">
                    <img
                      src={img}
                      alt={`Annotated page ${idx + 1}`}
                      className="annotated-image"
                    />
                    <span className="red-box-label">
                      {invoice.file_type === "PDF" ? `Page ${idx + 1}` : "Image"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    );
  };
  
 
  return (
    <div className="app-container">
      <nav className="navbar">
        <ul className="nav-links">
          <li>
            <a href="#">Home</a>
          </li>
          <li>
            <a href="#">Settings</a>
          </li>
          <li>
            <a href="#">Profile</a>
          </li>
        </ul>
        <div className="logo-container">
          <img
            src="https://listcarbrands.com/wp-content/uploads/2017/11/Bajaj-Logo.png"
            alt="Bajaj Logo"
            className="navbar-logo"
          />
        </div>
      </nav>

      <div className="title-container">
        <h1 className="app-title">Invoice Fraud Analyzer</h1>
        <p className="app-subtitle">Detect fraud and tampering in invoices with ease</p>
      </div>

      {view === "storage_selection" && (
        <div className="storage-selection">
          <div className="storage-card" onClick={() => handleStorageSelect("local")}>
            <img
              src="https://openclipart.org/image/2400px/svg_to_png/216996/1428616740.png"
              alt="Local Storage"
            />
            <h3>Local Storage</h3>
          </div>
          <div className="storage-card" onClick={() => handleStorageSelect("aws")}>
            <img
              src="https://www.pngplay.com/wp-content/uploads/3/Amazon-Web-Services-AWS-Logo-Transparent-PNG.png"
              alt="AWS S3"
            />
            <h3>AWS S3</h3>
          </div>
        </div>
      )}

      {view === "local_upload" && (
        <div className="upload-section">
          <label htmlFor="fileUpload" className="upload-label">
            Select Files (Images or PDFs):
          </label>
          <input
            type="file"
            id="fileUpload"
            multiple
            accept=".jpg,.jpeg,.png,.pdf"
            onChange={handleFileChange}
            className="upload-input"
          />
          <button
            onClick={handleAnalyzeFraud}
            disabled={loading || !files || files.length === 0}
            className="analyze-button"
          >
            {loading ? "Analyzing..." : "Analyze for Fraud"}
          </button>
        </div>
      )}

      {view === "aws_buckets" && (
        <div className="buckets-section">
          <h2>Select a Bucket</h2>
          {buckets.length > 0 ? (
            <ul>
              {buckets.map((bucket) => (
                <li key={bucket} onClick={() => handleBucketSelect(bucket)}>
                  <span>{bucket}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p>No buckets found.</p>
          )}
        </div>
      )}

      {view === "aws_files" && (
        <div className="files-section">
          <h2>
            Select Files from <span>{selectedBucket}</span>
          </h2>

          {filesInBucket.length > 0 ? (
            <>
              <div className="files-counter">
                {selectedFiles.length} of {filesInBucket.length} files selected
              </div>
              <div className="file-items-container">
                {filesInBucket.map((file) => (
                  <div key={file} className="file-item">
                    <input
                      type="checkbox"
                      checked={selectedFiles.includes(file)}
                      onChange={() => handleFileToggle(file)}
                    />
                    <span>{file}</span>
                  </div>
                ))}
              </div>
              <button
                onClick={handleAnalyzeFraudS3}
                disabled={loading || selectedFiles.length === 0}
                className="analyze-button"
              >
                {loading ? "Analyzing..." : "Analyze Selected Files"}
              </button>
            </>
          ) : (
            <p>No files found in this bucket.</p>
          )}
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {loading && (
        <div className="futuristic-overlay">
          <div className="spinner-container">
            <div className="spinner">
              <div className="spinner-inner"></div>
            </div>
            <div className="spinner-text">{loadingMessage}</div>
          </div>
        </div>
      )}

      {view === "results" && results && (
        <div className="results-container">
          {results.invoices.map((invoice, index) => (
            <div key={index} className="invoice-result">
              <h2 className="invoice-title">Invoice: {invoice.filename}</h2>
              {renderFraudDetection(invoice)}
            </div>
          ))}
          <button onClick={handleStartOver} className="start-over-button">
            Analyze Another Set
          </button>
        </div>
      )}
    </div>
  );
};

export default App;