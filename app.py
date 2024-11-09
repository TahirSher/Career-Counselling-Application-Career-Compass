import streamlit as st
import pandas as pd
from transformers import pipeline
import time

# Load datasets from CSV files
@st.cache_resource
def load_csv_datasets():
    jobs_data = pd.read_csv("job_descriptions.csv")
    courses_data = pd.read_csv("courses_data.csv")
    return jobs_data, courses_data

jobs_data, courses_data = load_csv_datasets()

# Constants
universities_url = "https://www.4icu.org/top-universities-world/"

# Initialize the text generation pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-large")

qa_pipeline = load_pipeline()

# Streamlit App Interface
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <h1 style="font-size: 29px; display: inline-block; margin-right: 10px;">
            <img src="https://img.icons8.com/ios-filled/50/000000/graduation-cap.png" width="40" alt="Degree icon"/>
            Confused about which career to pursue?
        </h1>
        <h2 style="font-size: 25px; display: inline-block; margin: 0;">Let CareerCompass help you decide in two simple steps</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display the appropriate subheader based on profile data status
if "profile_data" not in st.session_state or not st.session_state.get("profile_data_saved", False):
    st.markdown("<h3 style='font-size: 20px;'>Step 1: Find out profile questions on the left sidebar and follow the instructions.</h3>", unsafe_allow_html=True)

# Sidebar for Profile Setup
st.sidebar.header("Profile Setup")
educational_background = st.sidebar.selectbox("Educational Background", [
    "Computer Science", "Engineering", "Business Administration", "Life Sciences",
    "Social Sciences", "Arts and Humanities", "Mathematics", "Physical Sciences",
    "Law", "Education", "Medical Sciences", "Other"
])
interests = st.sidebar.text_input("Interests (e.g., AI, Data Science, Engineering)")
tech_skills = st.sidebar.text_area("Technical Skills (e.g., Python, SQL, Machine Learning)")
soft_skills = st.sidebar.text_area("Soft Skills (e.g., Communication, Teamwork)")

# Profile validation and saving
def are_profile_fields_filled():
    return all([educational_background, interests.strip(), tech_skills.strip(), soft_skills.strip()])

if st.sidebar.button("Save Profile"):
    if are_profile_fields_filled():
        with st.spinner('Saving your profile...'):
            time.sleep(2)
            st.session_state.profile_data = {
                "educational_background": educational_background,
                "interests": interests,
                "tech_skills": tech_skills,
                "soft_skills": soft_skills
            }
            st.session_state.profile_data_saved = True  # Set the profile data saved flag
            st.session_state.question_index = 0  # Initialize question index
            st.session_state.answers = {}  # Initialize dictionary for answers
            st.session_state.ask_additional_questions = None  # Reset question flag
            st.session_state.show_additional_question_buttons = True  # Show buttons after profile save
            st.sidebar.success("Profile saved successfully!")
            st.markdown("<h2 style='font-size: 25px;'>Step 2: For more Accurate Analysis, Do you wish to provide more information?</h2>", unsafe_allow_html=True)
    else:
        st.sidebar.error("Please fill in all the fields before saving your profile.")

# Button actions
if "show_additional_question_buttons" in st.session_state:
    if st.session_state.show_additional_question_buttons:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, ask me more questions"):
                st.session_state.ask_additional_questions = True
                st.session_state.show_additional_question_buttons = False  # Hide buttons after click
        with col2:
            if st.button("Skip and generate recommendations"):
                st.session_state.ask_additional_questions = False
                st.session_state.show_additional_question_buttons = False  # Hide buttons after click

# Additional questions for more tailored recommendations
additional_questions = [
    "What subjects do you enjoy learning about the most, and why?",
    "What activities or hobbies do you find most engaging and meaningful outside of school?",
    "Can you describe a perfect day in your dream career? What tasks would you be doing?",
    "Are you more inclined towards working independently or as part of a team?",
    "Do you prefer structured schedules or flexibility in your work?",
    "What values are most important to you in a career (e.g., creativity, stability, helping others)?",
    "How important is financial stability to you in your future career?",
    "Are you interested in pursuing a career that involves working with people, technology, or the environment?",
    "Would you prefer a career with a clear progression path or one with more entrepreneurial freedom?",
    "What problems or challenges do you want to solve or address through your career?"
]

# Display dynamic questions or proceed to generating recommendations
if "profile_data" in st.session_state:
    if st.session_state.get("ask_additional_questions") is True:
        total_questions = len(additional_questions)
        if "question_index" not in st.session_state:
            st.session_state.question_index = 0

        if st.session_state.question_index < total_questions:
            question_number = st.session_state.question_index + 1
            question = additional_questions[st.session_state.question_index]

            # Display question number and question text
            st.markdown(f"""### Question {question_number}:
             {question}""")

            answer = st.text_input("Your Answer", key=f"q{st.session_state.question_index}")

            # Display progress bar with formatted text showing "current/total"
            progress = (st.session_state.question_index + 1) / total_questions
            st.progress(progress)
            st.write(f"Progress: {question_number}/{total_questions}")

            if st.button("Submit Answer", key=f"submit{st.session_state.question_index}"):
                if answer:
                    st.warning("Data saved successfully. click again to proceed")
                    # Save the answer and increment the question index
                    st.session_state.question_index += 1
                    st.session_state.answers[question] = answer

                    # No need to call a special function; the app will rerun automatically
                else:
                    st.warning("Please enter an answer before submitting.")
        else:
            st.success("All questions have been answered. Click below to generate your recommendations.")
            if st.button("Generate Response"):
                st.warning("Data saved successfully. click again to proceed")
                st.session_state.profile_data.update(st.session_state.answers)
                st.session_state.ask_additional_questions = False

    elif st.session_state.get("ask_additional_questions") is False:
        # Directly generate recommendations
        st.header("Generating Recommendations")
        with st.spinner('Generating recommendations...'):
            time.sleep(2)  # Simulate processing time

            # Extracting user profile data
            profile = st.session_state.profile_data
            user_tech_skills = set(skill.strip().lower() for skill in profile["tech_skills"].split(","))
            user_soft_skills = set(skill.strip().lower() for skill in profile["soft_skills"].split(","))
            user_interests = set(interest.strip().lower() for interest in profile["interests"].split(","))
            user_answers = st.session_state.get('answers', {})

            # Job Recommendations using refined scoring logic
            def match_job_criteria(row, profile, user_answers):
                job_title = row['Job Title'].lower()
                job_description = row['Job Description'].lower()
                qualifications = row['Qualifications'].lower()
                skills = row['skills'].lower()
                role = row['Role'].lower()

                educational_background = profile['educational_background'].lower()
                tech_skills = set(skill.strip().lower() for skill in profile["tech_skills"].split(","))
                soft_skills = set(skill.strip().lower() for skill in profile["soft_skills"].split(","))
                interests = set(interest.strip().lower() for interest in profile["interests"].split(","))
                user_answers_text = ' '.join(user_answers.values()).lower()

                score = 0

                if educational_background in qualifications or educational_background in job_description:
                    score += 2
                if any(skill in skills for skill in tech_skills):
                    score += 3
                if any(skill in job_description or role for skill in soft_skills):
                    score += 1
                if any(interest in job_title or job_description for interest in interests):
                    score += 2
                if any(answer in job_description or qualifications for answer in user_answers_text.split()):
                    score += 2

                return score >= 5

            # Get unique job recommendations
            job_recommendations = jobs_data[jobs_data.apply(lambda row: match_job_criteria(row, profile, user_answers), axis=1)]
            unique_jobs = job_recommendations.drop_duplicates(subset=['Job Title'])

            # Display Job Recommendations in a table with bold job titles
            st.subheader("Job Recommendations")
            if not unique_jobs.empty:
                job_list = unique_jobs.head(5)[['Job Title', 'Job Description']].reset_index(drop=True)
                job_list['Job Title'] = job_list['Job Title'].apply(lambda x: f"<b>{x}</b>")
                job_list_html = job_list.to_html(index=False, escape=False, justify='left').replace(
                    '<th>', '<th style="text-align: left; font-weight: bold;">')
                st.markdown(job_list_html, unsafe_allow_html=True)
            else:
                st.write("No specific job recommendations found matching your profile.")
                st.write("Here are some general job recommendations:")
                fallback_jobs = jobs_data.drop_duplicates(subset=['Job Title']).head(3)
                fallback_jobs['Job Title'] = fallback_jobs['Job Title'].apply(lambda x: f"<b>{x}</b>")
                fallback_list_html = fallback_jobs[['Job Title', 'Job Description']].to_html(
                    index=False, escape=False, justify='left').replace(
                    '<th>', '<th style="text-align: left; font-weight: bold;">')
                st.markdown(fallback_list_html, unsafe_allow_html=True)

            # Course Recommendations using RAG technique
            course_recommendations = courses_data[courses_data['Course Name'].apply(
                lambda name: any(interest in name.lower() for interest in user_interests)
            )]

            # Display Course Recommendations
            st.subheader("Recommended Courses")
            if not course_recommendations.empty:
                for _, row in course_recommendations.head(5).iterrows():
                    st.write(f"- [{row['Course Name']}]({row['Links']})")
            else:
                st.write("No specific course recommendations found matching your interests.")
                st.write("Here are some general course recommendations aligned with your profile:")

                fallback_courses = courses_data[
                    courses_data['Course Name'].apply(
                        lambda name: any(
                            word in name.lower() for word in profile["educational_background"].lower().split() +
                            [skill.lower() for skill in profile["tech_skills"].split(",")]
                        )
                    )
                ]

                if not fallback_courses.empty:
                    for _, row in fallback_courses.head(3).iterrows():
                        st.write(f"- [{row['Course Name']}]({row['Links']})")
                else:
                    st.write("Consider exploring courses in fields related to your educational background or technical skills.")

# University Recommendations Section
st.header("Top Universities")
st.write("For further education, you can explore the top universities worldwide:")
st.write(f"[View Top Universities Rankings]({universities_url})")

st.write("Thank you for using the Career Counseling Application with RAG!")