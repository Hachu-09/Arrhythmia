import pickle
import numpy as np
import google.generativeai as gemini


gemini.configure(api_key="AIzaSyBoWR82ABdhXR4HNTAQsFJ0wMdaJQkeesY")


with open('HEART_MODEL/Arrhythmia/arrhythmia_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)


def make_prediction(input_data):
    
    prediction = ensemble_model.predict(np.array(input_data).reshape(1, -1))[0]
    probability = ensemble_model.predict_proba(np.array(input_data).reshape(1, -1))[0][1]
    return prediction, probability


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
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk:.2f}%
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        
        print("Generating report with the following prompt:")
        print(prompt)

     
        response = gemini.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1000
        )

        # Debugging: Print the response object to check its structure
        print("API Response:", response)

        if hasattr(response, 'result'):
            report = response.result
        else:
            print("The response from the API did not contain a result.")
            report = None
        
        return report
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# Example interactive input
print("\n--- Predict New Input ---")
input_data = []

# Collect input data for the specified features
try:
    age = float(input("Enter age: "))
    sex = float(input("Enter sex (0 for female, 1 for male): "))
    height = float(input("Enter height (in cm): "))
    weight = float(input("Enter weight (in kg): "))
    qrs_duration = float(input("Enter QRS duration (in ms): "))
    p_r_interval = float(input("Enter P-R interval (in ms): "))
    q_t_interval = float(input("Enter Q-T interval (in ms): "))
    t_interval = float(input("Enter T interval (in ms): "))
    p_interval = float(input("Enter P interval (in ms): "))
    qrs = float(input("Enter QRS complex value: "))
    QRST = float(input("Enter QRST value: "))
    heart_rate = float(input("Enter heart rate (in bpm): "))
    q_wave = float(input("Enter Q wave value: "))
    r_wave = float(input("Enter R wave value: "))
    s_wave = float(input("Enter S wave value: "))
    R_prime_wave = float(input("Enter R' wave value: "))
    S_prime_wave = float(input("Enter S' wave value: "))

    # Add input values to the list in order
    input_data.extend([age, sex, height, weight, qrs_duration, p_r_interval, q_t_interval,
                       t_interval, p_interval, qrs, QRST, heart_rate, q_wave, r_wave,
                       s_wave, R_prime_wave, S_prime_wave])

    # Making prediction based on user input
    prediction, prediction_prob = make_prediction(input_data)

    # Calculate risk percentage
    risk_percentage = prediction_prob * 100 if prediction == 1 else 0

    # Determine the type of disease based on prediction
    disease_type = "Arrhythmia" if prediction == 1 else "No Arrhythmia"

    # Display results
    print(f"Risk: {'You are at high risk of arrhythmia' if prediction == 1 else 'You are not having arrhythmia'}")
    print(f"Risk Percentage: {risk_percentage:.2f}%")
    print(f"Problem: {disease_type}")

    # Generate the wellness report using the risk and disease information if arrhythmia is detected
    if prediction == 1:
        report = generate_prevention_report(
            risk=risk_percentage,
            disease=disease_type,
            age=age
        )

        if report:
            print("\nGenerated Wellness Report:")
            print(report)
        else:
            print("Failed to generate a report. Please check the API response and try again.")

except Exception as e:
    print(f"An error occurred during prediction: {e}")
