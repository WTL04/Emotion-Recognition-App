# Goal 
- Create a CNN supervised learning model that can detect emotions on a person's face in real-time (webcam)
- Create a full stack application that hosts the model and has a nice UI/UX 

# Problem that it solves 
Problem:
- Businesses want to understand how customers feel about their products or services, but traditional surveys and feedback forms can be slow and subjective.

Solution:
- Use facial emotion detection to analyze customers’ emotional reactions in real-time during
product testing, customer service interactions, or while engaging with a website. This data can be used to gauge satisfaction levels and improve user experiences.

# Tech Stack 

Dataset 
- FER-2013 Dataset (has labeled images of faces with emotions)

Libraries
- Pytorch: making the model 
- OpenCV: for real-time CV 
- Numpy/Pandas: for data manipulation 
- Intel oneAPI: optional, uses Intel graphics to accelerate computation

Tech Stack:
- Frontend: Nextjs and Tailwindcss 
- Backend: FastAPI 

# Progress 
04/04/2025 
- Model accuracy up to roughly 70%
- Confusion matrix states that the model struggles with "Anger"
- "Sad" is also confused with "Neutral"
