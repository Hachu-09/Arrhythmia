from flask import Flask, request, jsonify
import pickle
import numpy as np
import google.generativeai as gemini
import google.generativeai as genai


# Configure the Gemma API
genai.configure(api_key="AIzaSyASUFBrNl_EsBuo8QD2_1HDGZXlcVAiG_o")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the Flask application
app = Flask(__name__)

# Configure the Gemini API


# Load the trained model from the pickle file
with open('HEART_MODEL/Arrhythmia/arrhythmia_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Function to make a prediction with new data
def make_prediction(input_data):
    # Convert the dictionary of input features into a list of values
    input_values = list(input_data.values())
    
    # Ensure all inputs are numerical (floats)
    input_array = np.array(input_values, dtype=float).reshape(1, -1)
    
    # Predict the result and the probability
    prediction = ensemble_model.predict(input_array)[0]
    probability = ensemble_model.predict_proba(input_array)[0][1]
    return prediction, probability

# Function to generate a prevention report based on risk and disease
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        -Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it
    Risk: {risk:.2f}%
    Disease: {disease}
    Age: {age}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)

    # Print received data for debugging
    print("Received data:", data)

    # Check if the data is in the expected format
    if not data or 'input' not in data:
        return jsonify({'error': 'No input data provided'}), 400

    input_data = data.get('input')
    
    try:
        # Ensure that input_data is not empty and contains the right number of features
        if not input_data or len(input_data) == 0:
            return jsonify({'error': 'Input data is empty or invalid'}), 400

        # Make a prediction
        prediction, probability = make_prediction(input_data)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # Generate the prevention report
    disease = "Arrhythmia"
    age = data.get('age', 'Unknown')  # Use 'Unknown' if age is not provided
    
    prevention_report = generate_prevention_report(probability, disease, age)
    
    # Prepare the result
    result = {
        "prediction": int(prediction),
        "probability": float(probability),
        "prevention_report": prevention_report
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
