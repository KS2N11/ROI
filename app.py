import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins in production

# Frontend configuration modification
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://roi-3dqo.onrender.com')

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')

try:
    client = AzureOpenAI(  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION,
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    client = None

@app.route('/generate-ai-observations', methods=['POST'])
def generate_ai_observations():
    try:
        # Validate OpenAI client is initialized
        if client is None:
            return jsonify({"error": "OpenAI client not initialized"}), 500

        data = request.json.get("forecastData", [])

        if not data:
            return jsonify({"error": "No forecast data provided"}), 400

        # Validate data structure
        required_keys = ['year', 'investment', 'amcCost', 'revenue', 'savings', 
                         'totalRevenueSavings', 'cumulativeTotalCost', 
                         'netProfitLoss', 'cumulativeProfitLoss']
        
        for entry in data:
            for key in required_keys:
                if key not in entry:
                    return jsonify({"error": f"Missing required key: {key}"}), 400

        # Format forecast data
        forecast_summary = "\n".join([
            (f"Year {entry['year']}: Investment = ${entry['investment']:.2f}, AMC Cost = ${entry['amcCost']:.2f}, "
             f"Revenue = ${entry['revenue']:.2f}, Cost Savings = ${entry['savings']:.2f}, Total Revenue & Savings = ${entry['totalRevenueSavings']:.2f}, "
             f"Cumulative Cost = ${entry['cumulativeTotalCost']:.2f}, Net Profit/Loss = ${entry['netProfitLoss']:.2f}, "
             f"Cumulative Profit/Loss = ${entry['cumulativeProfitLoss']:.2f}")
            for entry in data
        ])

        prompt = (
    "You are a financial analyst with expertise in return on investment (ROI) forecasting. "
    "Analyze the following financial forecast and provide **concise, data-driven** insights. "
    "Each point must be **a maximum of four lines**, covering all relevant data from the forecast. "
    "Insights should include **both quarterly and yearly trends**, backed by **specific numerical values**. "
    "Ensure the response follows the structured format below.\n\n"
    "#### **Financial Data:**\n"
    f"{forecast_summary}\n\n"
    "#### **Your Analysis Should Cover:**\n"
    "Provide exactly **4 key insights** and **1 two-line conclusion** (always include ROI Trends and Break-even Point) in the following structured format:\n\n"
    "---\n"
    "## 📊 **Key Insights**\n\n"
    "- **Quarterly ROI growth** starts at **2.5% in Q1**, rising to **7% in Q4**, leading to an annual ROI of **28%** in Year 1. "
    "By Year 3, **quarterly ROI stabilizes at 9%**, pushing the annual ROI to **120%**.\n\n"
    "- **Break-even is reached in Q2 of Year 2**, when revenue crosses **$450,000**, surpassing cumulative costs of **$380,000**. "
    "By Year 3, net profit reaches **$150,000 per quarter**, ensuring long-term stability.\n\n"
    "- **Compared to industry benchmarks**, projected ROI exceeds the **95% 3-year industry average** by **15%**, "
    "with **quarterly cost efficiency improving by 5% per quarter**, strengthening profit margins.\n\n"
    "- **High Year 1 operational costs** of **$100,000 per quarter** exceed projections by **20%**, but automation reduces "
    "quarterly expenses by **$10,000 from Q3 onward**, leading to **total savings of $200,000 over 5 years**.\n\n"
    "---\n"
    "## 📌 **Conclusion**\n"
    "💡 **ROI reaches 120% in 3 years, with quarterly gains stabilizing at 9%. "
    "Break-even in Q2 of Year 2 ensures sustainable profits, while automation-driven savings boost cost efficiency long-term.**"
)



        # Add error handling for OpenAI API call
        try:
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_observation = response.choices[0].message.content.strip()
        except Exception as openai_error:
            print(f"OpenAI API Error: {openai_error}")
            return jsonify({"error": f"OpenAI API Error: {str(openai_error)}"}), 500

        return jsonify({"observation": ai_observation})

    except Exception as e:
        # Log the full stack trace
        print(f"Error in AI Observations API:")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
