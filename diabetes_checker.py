# diabetes_checker.py - Simplified version without training dependencies
import joblib
import numpy as np

class DiabetesChecker:
    def __init__(self):
        # Load the trained model and scaler
        self.model = joblib.load("models/diabetes_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
    
    def get_user_input(self):
        print("=== Diabetes Risk Assessment ===")
        print("Please enter the following patient information:\n")
        
        data = {}
        data['Pregnancies'] = int(input("Number of pregnancies: "))
        data['Glucose'] = float(input("Glucose level (mg/dL): "))
        data['BloodPressure'] = float(input("Blood pressure (mmHg): "))
        data['SkinThickness'] = float(input("Skin thickness (mm): "))
        data['Insulin'] = float(input("Insulin level (mu U/ml): "))
        data['BMI'] = float(input("BMI (kg/m²): "))
        data['DiabetesPedigreeFunction'] = float(input("Diabetes pedigree function: "))
        data['Age'] = int(input("Age: "))
        
        return data
    
    def preprocess_input(self, input_data):
        features = np.array([[
            input_data['Pregnancies'],
            input_data['Glucose'],
            input_data['BloodPressure'],
            input_data['SkinThickness'],
            input_data['Insulin'],
            input_data['BMI'],
            input_data['DiabetesPedigreeFunction'],
            input_data['Age']
        ]])
        
        scaled_features = self.scaler.transform(features)
        return scaled_features
    
    def predict(self, input_data):
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0][1]
        return prediction, probability
    
    def interpret_result(self, prediction, probability):
        if prediction == 1:
            result = "HIGH RISK of diabetes"
            advice = "Please consult a doctor for further evaluation and management."
        else:
            result = "LOW RISK of diabetes"
            advice = "Maintain healthy lifestyle with regular check-ups."
        return result, advice
    
    def run(self):
        try:
            user_data = self.get_user_input()
            prediction, probability = self.predict(user_data)
            result, advice = self.interpret_result(prediction, probability)
            
            print("\n" + "="*50)
            print("DIABETES RISK ASSESSMENT RESULTS")
            print("="*50)
            print(f"Prediction: {result}")
            print(f"Probability of diabetes: {probability:.2%}")
            print(f"Advice: {advice}")
            print("="*50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please make sure you entered valid numbers.")

if __name__ == "__main__":
    checker = DiabetesChecker()
    checker.run()