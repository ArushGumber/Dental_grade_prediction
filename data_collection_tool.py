import streamlit as st
import pandas as pd
import numpy as np
import io
from fpdf import FPDF 
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

st.set_page_config(
    page_title="IOTN Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
def predict_IOTN(patient_parameter_list):
    # The parameter order is assumed to be:
    # ['sample', 'age', 'sex', 'cleft', 'hypdonta',
    #  'overjet', 'reverseoverjet', 'crossbite',
    #  'displacement', 'openbite', 'overbite']
    patient_data = {}
    parameter_names = [
        'sample', 'age', 'sex', 'cleft', 'hypdonta',
        'overjet', 'reverseoverjet', 'crossbite',
        'displacement', 'openbite', 'overbite'
    ]
    for i, parameter in enumerate(parameter_names):
        patient_data[parameter] = float(patient_parameter_list[i])
    grade = 1

    # Overjet (Measurable)
    if 3.5 < patient_data['overjet'] <= 6 and grade < 2:
        grade = 3
    elif 6 < patient_data['overjet'] <= 9 and grade < 4:
        grade = 4
    elif patient_data['overjet'] > 9 and grade < 5:
        grade = 5

    # Reverse Overjet (Measurable)
    if 0 < patient_data['reverseoverjet'] <= 1 and grade < 2:
        grade = 2
    elif 1 < patient_data['reverseoverjet'] <= 3.5 and grade < 4:
        grade = 3
    elif patient_data['reverseoverjet'] > 3.5 and grade < 5:
        grade = 5

    # Hypodontia (and cleft+hypodontia)
    if patient_data['hypdonta'] > 0 and grade < 4:
        grade = 4
    elif patient_data['cleft'] > 0 and patient_data['hypdonta'] > 0 and grade < 5:
        grade = 5

    # Crossbite
    if 0 < patient_data['crossbite'] <= 1 and grade < 2:
        grade = 2
    elif 1 < patient_data['crossbite'] <= 2 and grade < 3:
        grade = 3
    elif patient_data['crossbite'] > 2 and grade < 4:
        grade = 4

    # Displacement
    if 1 < patient_data['displacement'] <= 2 and grade < 2:
        grade = 2
    elif 2 < patient_data['displacement'] <= 4 and grade < 3:
        grade = 3
    elif patient_data['displacement'] > 4 and grade < 4:
        grade = 4

    # Openbite
    if 1 < patient_data['openbite'] <= 2 and grade < 3:
        grade = 2
    elif 2 < patient_data['openbite'] <= 4 and grade < 4:
        grade = 3
    elif patient_data['openbite'] > 4 and grade < 5:
        grade = 4

    # Overbite
    if 3.5 < patient_data['overbite'] <= 6 and grade < 2:
        grade = 2

    return grade

custom_css = """
<style>
body {
    margin: 0;
    background-color: white;
    font-family: "monospace", monospace;
}
a {
    text-decoration: none; 
    font-weight: bold;      
    color: inherit;       
    underline: none;  
}
.header {
    width: 100%;
    font: monospace;
    background-color: white;
    border-top: 4px solid #ADD8E6;
    border-bottom: 4px solid #ADD8E6;
    padding: 20px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 80px;
    margin-top: 0px; 
}
.header-left, .header-center, .header-right {
    flex: 1;
}
.header-center {
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.header-right {
    text-align: right;
}
.header-left img, .header-right img {
    max-height: 60px;
    width: auto;
}
.footer {
    width: 100%;
    font: monospace;
    background-color: white;
    border-top: 4px solid #ADD8E6;
    border-bottom: 4px solid #ADD8E6;
    padding: 20px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 80px;
    margin-top: 20px;
}
.footer-center {
    text-align: center;
    flex: 1;
}
.footer-center .developed {
    color: black;
    font-weight: bold;
}
.footer-center .advisors {
    color: red;
}
.footer-right {
    text-align: right;
}
.footer-right img {
    max-height: 80px;
    width: auto;
}
</style>
"""

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
logo_base64 = get_base64_encoded_image("assets/logo/logo.png")
sbilogo_base64 = get_base64_encoded_image("assets/logo/sbilogo.png")

mamclogo_base64 = get_base64_encoded_image("assets/logo/mamc-logo.png")

header_html = f"""
<div class="header">
    <div class="header-left">
        <a href="https://www.iiitd.ac.in/" target="_blank">
            <img src="data:image/png;base64,{logo_base64}" alt="IIITD Logo">
        </a>
        <a href="https://sbilab.iiitd.edu.in/" target="_blank">
            <img src="data:image/png;base64,{sbilogo_base64}" alt="SBI Lab Logo">
        </a>
    </div>
    <div class="header-center">
        In Collaboration With
    </div>
    <div class="header-right">
        <a href="https://mamc.delhi.gov.in/" target="_blank">
            <img src="data:image/png;base64,{mamclogo_base64}" alt="MAMC Logo">
        </a>
    </div>
</div>
"""

footer_html = f"""
<div class="footer">
    <div class="footer-center">
        <div class="advisors">
            PIs: Sr. Prof. Dr. Tulika Tripathi (MAMC); <a href="https://www.iiitd.ac.in/anubha" target="_blank">Dr. Anubha Gupta</a> (IIITD)
        </div>
        <div class="developed">
            Team: <a href="https://www.linkedin.com/in/arushgumber/" target="_blank">Arush Gumber (IIITD)</a>; 
            Dr. Peemit Rawat (MAMC); Dr. Rinkle (MAMC)
        </div>
    </div>
    <div class="footer-right">
        <a href="https://sbilab.iiitd.edu.in/" target="_blank">
            <img src="data:image/png;base64,{sbilogo_base64}" alt="SBI Lab Logo">
        </a>
    </div>
</div>
"""


def multiple_patient_page():
    st.header("Multiple Patient Predictions (CSV Input)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        st.subheader("Edit the CSV file if needed:")
        edited_df = st.data_editor(df, key="data_editor", num_rows="dynamic")
        edited_df = edited_df.astype(str).T
        df = edited_df.copy()
        num_incorrect_preds = 0
        if st.button("Get Predictions"):
            df.loc[df.shape[0]] = [""] * df.shape[1]
            df.iloc[-1, 0] = 'Predicted Grade'
            for col in range(1, df.shape[1]):
                try:
                    patient_data_list = df.iloc[0:11, col].astype(float).tolist()
                    predicted_grade = predict_IOTN(patient_data_list)
                    df.iloc[-1, col] = predicted_grade
                except Exception as e:
                    st.error(f"Error processing column {col}: {e}")
            for i in range(1, df.shape[1]):
                try:
                    if int(df.iloc[-2, i]) != int(df.iloc[-1, i]):
                        st.write(f"Incorrect prediction at index {i} ; predicted: {df.iloc[-1, i]} ; actual: {df.iloc[-2, i]}")
                        num_incorrect_preds += 1
                except Exception as e:
                    st.write(f"Could not compare index {i}: {e}")
            st.write(f"Total Incorrect Predictions: {num_incorrect_preds}")
            df = df.T
            st.subheader("Final CSV with Predictions:")
            st.dataframe(df)
            
            csv = df.to_csv(index=False, header=False)
            st.download_button(
                label="Download CSV with Predictions",
                data=csv,
                file_name='predicted_IOTN.csv',
                mime='text/csv'
            )
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, header=False)
            excel_data = output.getvalue()
            st.download_button(
                label="Download Excel with Predictions",
                data=excel_data,
                file_name='predicted_IOTN.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

def individual_patient_page():
    st.header("Individual Patient Input and Report Generation")
    st.write("Enter the patient details below:")
    
    sample = st.number_input("Patient ID", min_value=1, value=1)
    age = st.number_input("Age", min_value=0, value=18)
    sex = st.selectbox("Sex (0=Male, 1=Female)", options=[0, 1],
                       format_func=lambda x: "Male" if x == 0 else "Female")
    cleft = st.selectbox("Cleft lip/palate (0=No, 1=Yes)", options=[0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes")
    hypdonta = st.selectbox("Hypodontia (0=No, 1=Yes)", options=[0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
    overjet = st.number_input("Overjet (mm)", min_value=0.0, value=5.0, step=0.1, format="%.1f")
    reverseoverjet = st.number_input("Reverse Overjet (mm)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    crossbite = st.number_input("Crossbite (mm)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    displacement = st.number_input("Displacement (mm)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
    openbite = st.number_input("Openbite (mm)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    overbite = st.number_input("Overbite (mm)", min_value=0.0, value=4.0, step=0.1, format="%.1f")
    
    if st.button("Predict IOTN Grade"):
        patient_data_list = [
            sample, age, sex, cleft, hypdonta,
            overjet, reverseoverjet, crossbite,
            displacement, openbite, overbite
        ]
        predicted_grade = predict_IOTN(patient_data_list)
        st.success(f"Predicted IOTN Grade: {predicted_grade}")
        
        factors = {}
        if overjet > 9:
            factors["Overjet"] = (5, f"An overjet of {overjet} mm exceeds 9 mm, indicating a severe discrepancy.")
        elif 6 < overjet <= 9:
            factors["Overjet"] = (4, f"An overjet of {overjet} mm falls between 6 and 9 mm, suggesting a moderate discrepancy.")
        elif 3.5 < overjet <= 6:
            factors["Overjet"] = (3, f"An overjet of {overjet} mm indicates a mild increase.")
        
        if reverseoverjet > 3.5:
            factors["Reverse Overjet"] = (5, f"A reverse overjet of {reverseoverjet} mm is significantly high, indicating severe reversal.")
        elif 1 < reverseoverjet <= 3.5:
            factors["Reverse Overjet"] = (3, f"A reverse overjet of {reverseoverjet} mm suggests a moderate reversal.")
        elif 0 < reverseoverjet <= 1:
            factors["Reverse Overjet"] = (2, f"A reverse overjet of {reverseoverjet} mm is minimal.")
        
        if cleft > 0 and hypdonta > 0:
            factors["Cleft & Hypodontia"] = (5, "The presence of both a cleft and hypodontia indicates a severe dental anomaly.")
        elif hypdonta > 0:
            factors["Hypodontia"] = (4, "The presence of hypodontia is significant.")
        
        if crossbite > 2:
            factors["Crossbite"] = (4, f"A crossbite measuring {crossbite} mm indicates a significant occlusal discrepancy.")
        elif 1 < crossbite <= 2:
            factors["Crossbite"] = (3, f"A crossbite of {crossbite} mm suggests a moderate condition.")
        elif 0 < crossbite <= 1:
            factors["Crossbite"] = (2, f"A crossbite of {crossbite} mm is minimal.")
        
        if displacement > 4:
            factors["Displacement"] = (4, f"A displacement of {displacement} mm is severe and contributes markedly to misalignment.")
        elif 2 < displacement <= 4:
            factors["Displacement"] = (3, f"A displacement of {displacement} mm indicates moderate misalignment.")
        elif 1 < displacement <= 2:
            factors["Displacement"] = (2, f"A displacement of {displacement} mm is relatively minor.")
        
        if openbite > 4:
            factors["Openbite"] = (4, f"An openbite of {openbite} mm is severe.")
        elif 2 < openbite <= 4:
            factors["Openbite"] = (3, f"An openbite of {openbite} mm indicates a moderate condition.")
        elif 1 < openbite <= 2:
            factors["Openbite"] = (2, f"An openbite of {openbite} mm is minimal.")
        
        if overbite > 3.5 and overbite <= 6:
            factors["Overbite"] = (2, f"An overbite of {overbite} mm is slightly increased.")
        
        deciding_factor = None
        max_factor_grade = 1
        explanation_text = ""
        for factor, (grade_val, exp_text) in factors.items():
            if grade_val > max_factor_grade:
                max_factor_grade = grade_val
                deciding_factor = factor
                explanation_text = exp_text
        if deciding_factor is None:
            deciding_factor = "No single parameter was markedly abnormal"
            explanation_text = "All measurements are within acceptable ranges."
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "IOTN Prediction Report", ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Patient Details:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Patient ID: {sample}", ln=True)
        pdf.cell(0, 10, f"Age: {age}", ln=True)
        pdf.cell(0, 10, f"Sex: {'Male' if sex == 0 else 'Female'}", ln=True)
        pdf.cell(0, 10, f"Cleft Lip/Palate: {'Yes' if cleft == 1 else 'No'}", ln=True)
        pdf.cell(0, 10, f"Hypodontia: {'Yes' if hypdonta == 1 else 'No'}", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Dental Measurements:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Overjet: {overjet} mm", ln=True)
        pdf.cell(0, 10, f"Reverse Overjet: {reverseoverjet} mm", ln=True)
        pdf.cell(0, 10, f"Crossbite: {crossbite} mm", ln=True)
        pdf.cell(0, 10, f"Displacement: {displacement} mm", ln=True)
        pdf.cell(0, 10, f"Openbite: {openbite} mm", ln=True)
        pdf.cell(0, 10, f"Overbite: {overbite} mm", ln=True)
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Predicted IOTN Grade: {predicted_grade}", ln=True)
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analysis and Explanation:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, f"The most significant factor influencing the treatment need is '{deciding_factor}'. {explanation_text}")
        
        pdf.ln(10)
        pdf.multi_cell(0, 10, "This comprehensive report evaluates the dental parameters based on the IOTN system. "
                               "The identified abnormality plays a critical role in determining the severity of the condition. "
                               "It is strongly advised to consult a dental specialist for a thorough clinical evaluation and to discuss appropriate treatment options. "
                               "Please note that this report is for informational purposes only and does not replace professional diagnosis.")
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(
            label="Download Detailed PDF Report",
            data=pdf_bytes,
            file_name="IOTN_Prediction_Report.pdf",
            mime="application/pdf"
        )
        # import openai
        # openai.api_key = " "

        def generate_dental_report(patient_data, predicted_grade):
            prompt = f"""
        Generate a detailed dental report for the following patient details:
        Patient ID: {patient_data['sample']}
        Age: {patient_data['age']}
        Sex: {"Male" if patient_data['sex'] == 0 else "Female"}
        Cleft lip/palate: {"Yes" if patient_data['cleft'] else "No"}
        Hypodontia: {"Yes" if patient_data['hypdonta'] else "No"}
        Overjet: {patient_data['overjet']} mm
        Reverse Overjet: {patient_data['reverseoverjet']} mm
        Crossbite: {patient_data['crossbite']} mm
        Displacement: {patient_data['displacement']} mm
        Openbite: {patient_data['openbite']} mm
        Overbite: {patient_data['overbite']} mm

        The predicted IOTN Grade is {predicted_grade}.

        Please write a comprehensive dental report that explains the significance of these measurements, discusses the severity of the malocclusion based on the IOTN system, and provides clear recommendations for further consultation or treatment.
        """
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #     {"role": "system", "content": "You are a dental expert generating comprehensive and detailed dental reports."},
            #     {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=500,
            #     temperature=0.7,
            # )
            # report_text = response['choices'][0]['message']['content'].strip()
            # return report_text

        # patient_data = {
        #     "sample": sample,
        #     "age": age,
        #     "sex": sex,
        #     "cleft": cleft,
        #     "hypdonta": hypdonta,
        #     "overjet": overjet,
        #     "reverseoverjet": reverseoverjet,
        #     "crossbite": crossbite,
        #     "displacement": displacement,
        #     "openbite": openbite,
        #     "overbite": overbite
        # }
        # report = generate_dental_report(patient_data, predicted_grade)
        # st.write(report)


def eda_page():
    st.header("Exploratory Data Analysis and Bias Check")
    st.write("Upload a CSV file containing patient records for EDA and bias analysis. "
             "The CSV must be in the same format as your sample (with a header row).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="eda_page")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        expected_params = ["Sample", "Age", "Sex", "cleft lip/ palate", "hypdonta", 
                           "Overjet (mm)", "Reverse Overjet (mm)", "Crossbite (mm)", 
                           "Displacement(mm)", "Openbite(mm)", "Overbite(mm)"]
        
        missing_cols = [col for col in expected_params if col not in df.columns]
        if missing_cols:
            st.error("The following expected columns are missing from the CSV: " + ", ".join(missing_cols))
            return
        
        for col in expected_params:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["Predicted Grade"] = df.apply(
            lambda row: predict_IOTN([row[p] for p in expected_params]), axis=1
        )
        
        st.subheader("Data with Predicted Grades")
        st.dataframe(df)
        small_figsize = (5, 3)
        sns.set(style="whitegrid")
        
        # plot 1 gender distribution piechart
        fig1, ax1 = plt.subplots(figsize=(small_figsize))
        gender_counts = df["Sex"].value_counts().sort_index()
        labels = ["Male" if x == 0 else "Female" for x in gender_counts.index]
        ax1.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("pastel"))
        ax1.set_title("Gender Distribution", fontsize=10)
        st.pyplot(fig1)
        
        # plt 2 age distribution histogram
        fig2, ax2 = plt.subplots(figsize=(small_figsize))
        ax2.hist(df["Age"], bins=10, color="skyblue", edgecolor="black")
        ax2.set_xlabel("Age", fontsize=9)
        ax2.set_ylabel("Frequency", fontsize=9)
        ax2.set_title("Age Distribution", fontsize=10)
        st.pyplot(fig2)
        
        # plot 3 predicted grade distribution Barchart
        fig3, ax3 = plt.subplots(figsize=(small_figsize))
        grade_counts = df["Predicted Grade"].value_counts().sort_index()
        ax3.bar(grade_counts.index.astype(str), grade_counts.values, color="lightgreen", edgecolor="black")
        ax3.set_xlabel("Predicted Grade", fontsize=9)
        ax3.set_ylabel("Count", fontsize=9)
        ax3.set_title("Predicted Grade Distribution", fontsize=10)
        st.pyplot(fig3)
        
        # Plot5 gender distribution across predicted grades grouped bar
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        grade_gender = pd.crosstab(df["Predicted Grade"], df["Sex"])
        grade_gender.columns = ["Male", "Female"]
        grade_gender.plot(kind="bar", ax=ax4, color=["steelblue", "salmon"])
        ax4.set_title("Gender across Predicted Grades", fontsize=10)
        ax4.set_xlabel("Predicted Grade", fontsize=9)
        ax4.set_ylabel("Count", fontsize=9)
        st.pyplot(fig4)
        
        # plt 5 age distribution across predicted grades
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="Predicted Grade", y="Age", data=df, ax=ax5, palette="pastel")
        ax5.set_title("Age vs Predicted Grade", fontsize=10)
        st.pyplot(fig5)
        
        # plot 6 scatter plot Age vs Predicted Grade 
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="Age", y="Predicted Grade", hue="Sex", palette="Set1", ax=ax6)
        ax6.set_title("Scatter: Age vs Predicted Grade", fontsize=10)
        st.pyplot(fig6)
        
        st.subheader("Group-wise Aggregated EDA Summary")
        eda_summary = df.groupby("Predicted Grade").agg(
            count=("Sample", "count"),
            mean_age=("Age", "mean"),
            male_count=("Sex", lambda x: (x == 0).sum()),
            female_count=("Sex", lambda x: (x == 1).sum())
        ).reset_index()
        st.dataframe(eda_summary)
        
        csv_eda = eda_summary.to_csv(index=False)
        st.download_button(
            label="Download Group-wise EDA Summary",
            data=csv_eda,
            file_name="group_wise_eda_summary.csv",
            mime="text/csv"
        )
        
        #pdf
        if st.button("Generate EDA PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "EDA & Bias Analysis Report", ln=True, align="C")
            pdf.ln(5)

            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "This report provides an exploratory data analysis (EDA) and bias check for the patient records. "
                                    "It includes distribution plots for gender, age, and predicted grades, as well as a group-wise aggregated summary.")
            pdf.ln(5)
            plots = [("Gender Distribution", fig1),
                     ("Age Distribution", fig2),
                     ("Predicted Grade Distribution", fig3),
                     ("Gender across Predicted Grades", fig4),
                     ("Age vs Predicted Grade (Box Plot)", fig5),
                     ("Scatter: Age vs Predicted Grade", fig6)]
            
            for title, fig in plots:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.savefig(tmpfile.name, bbox_inches="tight")
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, title, ln=True)
                    pdf.image(tmpfile.name, x=10, w=pdf.w - 20)
                    pdf.ln(5)
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Group-wise Aggregated Summary:", ln=True)
            pdf.set_font("Arial", "", 10)
            for index, row in eda_summary.iterrows():
                line = f"Grade {row['Predicted Grade']}: Count = {row['count']}, Mean Age = {row['mean_age']:.1f}, Male = {row['male_count']}, Female = {row['female_count']}"
                pdf.cell(0, 10, line, ln=True)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="Download EDA PDF Report",
                data=pdf_bytes,
                file_name="EDA_Bias_Analysis_Report.pdf",
                mime="application/pdf"
            )

        



def main():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: red;'>IOTN Prediction App</h1>", unsafe_allow_html=True)
    
    mode = st.sidebar.selectbox("Choose Mode:", ("Multiple Patient Predictions (CSV Input)", "Individual Patient Input and Report Generation" , "Exploratory Data Analysis and Bias Check"))
    if mode == "Multiple Patient Predictions (CSV Input)":
        multiple_patient_page()
    elif mode == "Individual Patient Input and Report Generation":
        individual_patient_page()
    elif mode == "Exploratory Data Analysis and Bias Check":
        eda_page()
    
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
