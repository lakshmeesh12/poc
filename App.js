import React, { useState } from 'react';
import { extractInvoices } from './api';
import './home.css';

const App = () => {
    const [files, setFiles] = useState(null);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setFiles(e.target.files);
        setResults(null);
        setError(null);
    };

    const handleAnalyzeFraud = async () => {
        if (!files || files.length === 0) {
            setError('Please select at least one file to analyze.');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const data = await extractInvoices(files);
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const renderFraudDetection = (invoice) => {
        const { data, csv_validation } = invoice;
        const fields = [
            "What is the chassis number?",
            "What is the engine number?",
            "What is the make?",
            "What is the model?",
            "What is the color?",
            "What is the customer name?",
        ];

        const isCsvValidated = csv_validation.status === "Validated" && csv_validation.primary_key;

        if (!isCsvValidated) {
            return (
                <div className="fraud-section">
                    <h3>Fraud Detection Results</h3>
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
                    <div className="summary warning">
                        Cannot find a matching chassis number in the database: {data["What is the chassis number?"]?.value || "Not Found"} does not match.
                    </div>
                </div>
            );
        }

        const mismatches = fields.filter((field) => !data[field]?.csv_validated);
        const hasFraud = mismatches.length > 0;

        return (
            <div className="fraud-section">
                <h3>Fraud Detection Results</h3>
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
                <div className={`summary ${hasFraud ? "error" : "success"}`}>
                    {hasFraud ? (
                        <>
                            <strong>Fraud Detected:</strong> Mismatching values - {mismatches.map((field) => field.replace("What is the ", "").replace("?", "")).join(", ")}
                        </>
                    ) : (
                        <>
                            <strong>No Fraud Detected:</strong> All values match with the database.
                        </>
                    )}
                </div>
            </div>
        );
    };

    const renderTamperingDetection = (invoice) => {
        const { tamper_detection } = invoice;
        const { tampering_detected, confidence, methods, details, annotated_image } = tamper_detection;
    
        const detectedMethods = details.filter((detail) => detail.tampering_detected);
    
        return (
            <div className="tamper-section">
                <h3>Tampering Detection Results</h3>
                <div className="tamper-status">
                    <span className={`status-icon ${tampering_detected ? "alert" : "safe"}`}>
                        {tampering_detected ? "⚠️" : "✔️"}
                    </span>
                    <div>
                        <h4>{tampering_detected ? "Tampering Detected" : "No Tampering Detected"}</h4>
                        <p>Confidence Score: {(confidence * 100).toFixed(2)}%</p>
                    </div>
                </div>
                {tampering_detected && annotated_image && (
                    <div className="annotated-image">
                        <h4>Suspicious Areas Highlighted</h4>
                        <img 
                            src={`data:image/jpeg;base64,${annotated_image}`} 
                            alt="Tampered Regions" 
                            style={{ maxWidth: '100%', border: '2px solid #e74c3c', borderRadius: '5px' }} 
                        />
                        <p style={{ fontStyle: 'italic' }}>
                            Red: Font Issues | Blue: Alignment Issues | Green: ELA Anomalies
                        </p>
                    </div>
                )}
                {tampering_detected && detectedMethods.length > 0 && (
                    <div className="tamper-details">
                        <h4>Detected Tampering Methods:</h4>
                        <ul>
                            {detectedMethods.map((detail, index) => (
                                <li key={index}>
                                    <strong>{detail.method.replace(/_/g, " ").toUpperCase()}:</strong> {detail.details} (Confidence: {(detail.confidence * 100).toFixed(2)}%)
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                <div className="tamper-all-methods">
                    <h4>All Analysis Methods:</h4>
                    <div className="method-grid">
                        {details.map((detail, index) => (
                            <div key={index} className={`method-card ${detail.tampering_detected ? "alert" : "safe"}`}>
                                <p><strong>{detail.method.replace(/_/g, " ").toUpperCase()}</strong></p>
                                <p>{detail.details}</p>
                                <p>Confidence: {(detail.confidence * 100).toFixed(2)}%</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="app-container">
            {/* Navbar */}
            <nav className="navbar">
                <ul className="nav-links">
                    <li><a href="#">Home</a></li>
                    <li><a href="#">Settings</a></li>
                    <li><a href="#">Profile</a></li>
                </ul>
                <div className="logo-container">
                    <img 
                        src="https://listcarbrands.com/wp-content/uploads/2017/11/Bajaj-Logo.png" 
                        alt="Bajaj Logo" 
                        className="navbar-logo" 
                    />
                </div>
            </nav>

            {/* Adjusted Title */}
            <div className="title-container">
                <h1 className="app-title">Invoice Fraud Analyzer</h1>
                <p className="app-subtitle">Detect fraud and tampering in invoices with ease</p>
            </div>

            {/* Upload Section */}
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
                    disabled={loading}
                    className="analyze-button"
                >
                    {loading ? 'Analyzing...' : 'Analyze for Fraud'}
                </button>
            </div>

            {error && (
                <div className="error-message">{error}</div>
            )}

            {/* Loading Overlay */}
            {loading && (
                <div className="futuristic-overlay">
                    <div className="spinner-container">
                        <div className="spinner">
                            <div className="spinner-inner"></div>
                        </div>
                        <div className="spinner-text">ANALYZING</div>
                    </div>
                </div>
            )}

            {results && results.invoices && results.invoices.length > 0 && (
                <div className="results-container">
                    {results.invoices.map((invoice, index) => (
                        <div key={index} className="invoice-result">
                            <h2 className="invoice-title">Invoice: {invoice.filename}</h2>
                            {renderFraudDetection(invoice)}
                            {renderTamperingDetection(invoice)}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default App;