// src/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Base URL for your FastAPI server

/**
 * Extract invoice data from the uploaded files
 * @param {FileList|Array} files - List of files to upload (images or PDFs)
 * @returns {Promise<Object>} - Raw API response with extracted invoice data
 */
export const extractInvoices = async (files) => {
    try {
        // Create FormData object to handle file uploads
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file); // Match the 'files' key expected by the API
        }

        const response = await axios.post(`${API_BASE_URL}/extract-invoices/`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data', // Required for file uploads
            },
        });
        return response.data; // Return the raw response
    } catch (error) {
        throw new Error(error.response ? error.response.data.error : error.message);
    }
};