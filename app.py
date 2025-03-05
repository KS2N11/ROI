import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

app = Flask(__name__, 
            static_folder='.',  # Serve from current directory
            static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Validate critical environment variables
REQUIRED_ENVS = [
    'AZURE_OPENAI_API_KEY', 
    'AZURE_OPENAI_ENDPOINT', 
    'AZURE_DEPLOYMENT_NAME', 
    'AZURE_API_VERSION'
]

# Check for missing environment variables
missing_envs = [env for env in REQUIRED_ENVS if not os.getenv(env)]
if missing_envs:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_envs)}")


# Azure OpenAI Configuration
try:
    client = AzureOpenAI(  
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),  
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_API_VERSION'),
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    client = None

# Add a route to serve the frontend HTML
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # If no path specified or path is add_gain3.html, serve the HTML file
    if path == "" or path == "add_gain3.html":
        return send_from_directory('.', 'add_gain3.html')
    
    # Serve other static files from the current directory
    return send_from_directory('.', path)

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
                model=os.getenv('AZURE_DEPLOYMENT_NAME'),
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

# Add a simple health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "ROI Calculator Backend is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
