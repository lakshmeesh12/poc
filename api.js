// src/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Base URL for your FastAPI server

export const extractInvoices = async (files) => {
    try {
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }
        const response = await axios.post(`${API_BASE_URL}/extract-invoices/`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response ? error.response.data.error : error.message);
    }
};

export const listBuckets = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/list-buckets/`);
        return response.data.buckets;
    } catch (error) {
        throw new Error(error.response ? error.response.data.error : error.message);
    }
};

export const listFilesInBucket = async (bucket) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/list-files/`, {
            params: { bucket }
        });
        return response.data.files;
    } catch (error) {
        throw new Error(error.response ? error.response.data.error : error.message);
    }
};

export const extractInvoicesS3 = async (bucket, fileKeys) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/extract-invoices-s3/`, {
            bucket,
            file_keys: fileKeys
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response ? error.response.data.error : error.message);
    }
};