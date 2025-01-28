import os
import re
import base64
import json
from collections import OrderedDict
import time
from io import BytesIO
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
from fuzzywuzzy import process

app = None
# Global variables to store the ICD data
disease_icd_dict = None

def create_app():
    global app
    app = Flask(__name__)
    CORS(app)

    # Configuration with your specified paths
    app.config['JSON_FILE_PATH'] = "./output (1).json"
    app.config['EXCEL_FILE_PATH'] = "./test.xlsx"
    app.config['ICD_CODE_JSON_PATH'] = "./icd_codes.json"
    app.config['API_KEY'] = "AIzaSyAeLhT2j6ucPaE41P2ejgiMb2OVmKRoJKA"
    app.config['ICD_CODE_PATH'] = "./ICD.xlsx"


    # Initialize ICD codes
    with app.app_context():
        initialize_data(app)

    register_routes(app)
    return app

def initialize_data(app):
    global disease_icd_dict
    disease_icd_dict = load_icd_codes(app.config['ICD_CODE_PATH'])

def load_icd_codes(icd_code_path):
    df = pd.read_excel(icd_code_path)
    df.columns = df.columns.str.strip().str.lower()
    return dict(zip(df['diseases'], df['icd10_codes']))

def get_icd_code(disease_name, threshold=80):
    global disease_icd_dict
    if disease_icd_dict is None:
        raise ValueError("ICD codes have not been loaded. Call load_icd_codes() first.")
    
    match = process.extractOne(disease_name, disease_icd_dict.keys())
    if match and match[1] >= threshold:
        return disease_icd_dict[match[0]]
    return None

def process_base64_image(base64_string: str) -> bytes:
    try:
        # Remove data URI prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Open image using PIL
        with Image.open(BytesIO(image_bytes)) as image:
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95, optimize=True)
            processed_bytes = buffered.getvalue()
            
            if len(processed_bytes) == 0:
                raise ValueError("Processed image is empty")
                
            return processed_bytes
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 string: {str(e)}")
    except IOError as e:
        raise IOError(f"Error processing image: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error processing image: {str(e)}")

def clean_text(text: str) -> str:
    # Existing cleaning rules
    text = re.sub(r'\b-\b', '-', text)
    text = re.sub(r"\b'\b|\B'\b|\b'\B", "", text)
    text = re.sub(r"\\(\w+)\\", r"\1", text)
    text = re.sub(r'\s\?\s', ' ', text)
    text = re.sub(r'\b(?:age|a\.g|ag|age\.)\b', 'age', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:on|o\.n|on\.|o)\b', 'on', text, flags=re.IGNORECASE)
    text = re.sub(r'/\s*\(per\)', '', text)
    text = re.sub(r'\bc/0\b|\bclo\b|\bc/on\b|\bcLo\b|\bcho\b', 'Complains of', text, flags=re.IGNORECASE)
    text = re.sub(r'to\s*\(telephone order\)', 'to', text, flags=re.IGNORECASE)
    text = re.sub(r'AGE\s*\(acute gastro enteritis\)', 'age', text, flags=re.IGNORECASE)
    
    # Handle arrow symbols
    text = text.replace("↑", "increased ")
    text = text.replace("↓", "decreased ")
    text = text.replace("→", "leads to ")
    
    return text

def expand_abbreviations(text: str) -> str:
    abbreviations = {
        r'\bHTN\b': 'Hypertension',
        r'\bDM\b': 'Diabetes Mellitus',
        r'\bCHF\b': 'Congestive Heart Failure',
        r'\bCOPD\b': 'Chronic Obstructive Pulmonary Disease',
        r'\bMI\b': 'Myocardial Infarction',
        r'\bRA\b': 'Rheumatoid Arthritis',
        r'\bCKD\b': 'Chronic Kidney Disease',
        r'\bDVT\b': 'Deep Vein Thrombosis',
        r'\bGERD\b': 'Gastroesophageal Reflux Disease',
        r'\bMS\b': 'Multiple Sclerosis',
        r'\bIVDD\b': 'Intervertebral Disc Disease',
        r'\bCAD\b': 'Coronary Artery Disease',
        r'\bUTI\b': 'Urinary Tract Infection',
        r'\bCVA\b': 'Cerebrovascular Accident',
        r'\bAFib\b': 'Atrial Fibrillation',
        r'\bOA\b': 'Osteoarthritis',
        r'\bIBS\b': 'Irritable Bowel Syndrome',
        r'\bGI\b': 'Gastrointestinal',
        r'\bPVD\b': 'Peripheral Vascular Disease',
        r'\bTIA\b': 'Transient Ischemic Attack',
    }
    
    for abbr, full_form in abbreviations.items():
        text = re.sub(abbr, full_form, text, flags=re.IGNORECASE)
    
    return text

def analyze_document_from_base64(base64_string: str) -> Dict[str, Any]:
    try:
        genai.configure(api_key=current_app.config['API_KEY'])

        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        image_data = process_base64_image(base64_string)
        image_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode('utf-8')}

        prompt_template = """
    Analyze this document image thoroughly. Extract and categorize information based on the following criteria:

    1. **Text Analysis**:
       - Identify all text in the image.
       - Detect any underlined text and specify which words/phrases are underlined.
       - Identify segmented text (text split into separate boxes or sections).
       - Detect if any text is cut off or partially visible.
       - Identify any crossed-out or strikethrough text.

    2. **Symbol Detection**:
       - Detect the presence of checkboxes and whether they are checked or unchecked.
       - Identify any stars (★) or asterisks (*) near text and specify their location (before or after the text).

    3. **Shape Analysis**:
       - Identify any shapes present in the document (circles, squares, triangles, etc.).
       - Describe the relationship between shapes and nearby text.

    4. **Medication Information (if applicable)**:
       - Extract all mentions of medications (e.g., med1, med2, med3, med4).
       - For each medication, provide details and any associated shapes or symbols.

    5. **Form Structure**:
       - Identify form fields, labels, and their corresponding values.
       - Detect any tables and describe their content.

    6. **Special Cases**:
       - Note any handwritten text and its location.
       - Identify any logos, stamps, or signatures.
       - Detect any QR codes or barcodes.

    7. **Image Quality Assessment**:
       - Evaluate the overall image quality (e.g., clear, blurry, skewed).
       - Note any issues like poor contrast, shadows, or reflections.

    8. **Provisional Diagnosis and Abbreviation Handling**:
       - Capture all unique values under "Provisional Diagnosis," using "||" to separate only unique entries. Avoid duplicating values.
       - Distinguish between disease names and medication names:
           - Place disease names under the "Provisional Diagnosis" field.
           - Include medication names in the "remark" section.
       - For abbreviations in "Provisional Diagnosis":
           - Treat abbreviations as case-insensitive and replace them with full forms based on context.
           - Preserve surrounding text format when replacing abbreviations.
           - Note ambiguous abbreviations in the "remark" section.

    9. **Forged Check and Conflict Detection**:
       - Detect if any values conflict, are incongruous, or appear to be forged. Set the "Forged" field to true if such conflicts are found, and provide reasoning in the "Reason" section.

    Return the results in the following JSON format, with populated keys at the top of the JSON structure. If any information doesn’t fit the predefined format, summarize it concisely in the "remark" section.
    If a document have Medicine Name or something like prescription write it in "Medicine_Recommended" sections.
    ```json
    {
        "Treating_Doctor_Name": "",
        "Treating_Doctor_Contact_Number": "",
        "Nature_Of_Illness_Disease": "",
        "Nature_Of_Illness_Presenting_Complaint": "",
        "Relevant_Critical_Findings": "",
        "Duration_Of_Present_Ailment_Days": "",
        "Date_Of_First_Consultation": "",
        "Past_History_Of_Present_Ailment": "",
        "Provisional_Diagnosis": "",
        "Provisional_Diagnosis_ICD10_Code": "",
        "Medicine_Recommended" : {},
        "Medical_Management": false,
        "Surgical_Management": false,
        "Intensive_Care": false,
        "Investigation": false,
        "Non_Allopathic_Treatment": false,
        "Route_Of_Drug_Administration": "",
        "Surgical_Details_Name_Of_Surgery": "",
        "Surgical_Details_ICD10_PCS_Code": "",
        "Other_Treatment_Details": "",
        "How_Did_Injury_Occur": "",
        "Is_RTA": false,
        "Date_Of_Injury": "",
        "Report_To_Police": false,
        "FIR_No": "",
        "Substance_Abuse": false,
        "Test_Conducted": false,
        "Test_Conducted_Report_Attached": false,
        "Maternity_G": false,
        "Maternity_P": false,
        "Maternity_L": false,
        "Maternity_A": false,
        "Expected_Date_Of_Delivery": "",
        "Forged": false,
        "Reason": "",
        "Additional_Information": {},
        "Remark": ""
    }
    ```

    Ensure that fields with actual values appear at the top of the JSON output. Leave any non-applicable sections as empty strings, lists, or `false` as appropriate.
"""

        prompt_parts = [prompt_template, image_part]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt_parts)
                if not response or not response.text:
                    raise ValueError("Empty response from model")

                response_text = response.text.strip()
                current_app.logger.debug(f"Raw response from model: {response_text}")

                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError("No valid JSON found in the response")

                json_content = json_match.group(1)
                cleaned_json = clean_text(json_content)
                current_app.logger.debug(f"Cleaned JSON: {cleaned_json}")

                parsed_json = json.loads(cleaned_json, object_pairs_hook=OrderedDict)

                # Process multiple values and keep only unique ones
                for key, value in parsed_json.items():
                    if isinstance(value, str) and '||' in value:
                        unique_values = list(OrderedDict.fromkeys(value.split('||')))
                        parsed_json[key] = '||'.join(unique_values)

                # Expand abbreviations in Provisional_Diagnosis
                if 'Provisional_Diagnosis' in parsed_json:
                    parsed_json['Provisional_Diagnosis'] = expand_abbreviations(parsed_json['Provisional_Diagnosis'])

                # Add ICD code lookup
                if 'Provisional_Diagnosis' in parsed_json:
                    diagnoses = parsed_json['Provisional_Diagnosis'].split('||')
                    
                    icd_codes = []
                    for diagnosis in diagnoses:
                        icd_code = get_icd_code(diagnosis.strip())
                        if icd_code:
                            icd_codes.append(icd_code)
                    
                    if icd_codes:
                        parsed_json['Provisional_Diagnosis_ICD10_Code'] = "||".join(OrderedDict.fromkeys(icd_codes))
                    else:
                        parsed_json['Provisional_Diagnosis_ICD10_Code'] = "No matching ICD code found"

                return parsed_json

            except Exception as e:
                current_app.logger.error(f"Error in analyzing document: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)

    except Exception as e:
        current_app.logger.error(f"Error in analyzing document: {str(e)}")
        raise Exception(f"Error in analyzing document: {str(e)}")

def update_excel_with_new_data(new_data: Dict[str, Any]) -> bool:
    try:
        excel_path = current_app.config['EXCEL_FILE_PATH']
        if os.path.exists(excel_path):
            existing_df = pd.read_excel(excel_path)
        else:
            existing_df = pd.DataFrame()

        new_df = pd.DataFrame([new_data])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)
        current_app.logger.info("Excel file updated successfully.")
        return True
    except Exception as e:
        current_app.logger.error(f"Error updating Excel: {str(e)}")
        return False

def register_routes(app):
    @app.route('/process_image', methods=['POST'])
    def process_image():
        try:
            current_app.logger.debug("Request Headers: %s", request.headers)
            current_app.logger.debug("Content-Type: %s", request.content_type)

            data = request.get_json()
            current_app.logger.debug("Received JSON data: %s", data)

            if not data or 'image_base64' not in data:
                current_app.logger.error("Error: No base64 image data provided in the request.")
                return jsonify({'error': 'No base64 image data provided'}), 400

            base64_string = data['image_base64']
            current_app.logger.info("Processing base64 image data")
            extracted_data = analyze_document_from_base64(base64_string)
            current_app.logger.debug(extracted_data)

            with open(current_app.config['JSON_FILE_PATH'], 'w', encoding='utf-8') as json_file:
                json.dump(extracted_data, json_file, indent=4)

            excel_update_success = update_excel_with_new_data(extracted_data)

            return jsonify(extracted_data)

        except Exception as e:
            current_app.logger.error(f"Error during image processing: {str(e)}")
            return jsonify({'error': str(e)}), 500

app = create_app()
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
