from db import commit_submission
from TextProcessing import the_magic, submit_to_claimbust

import os
import re
import time
from fastapi import FastAPI, status, HTTPException, Header, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from twilio.rest import Client
from dotenv import load_dotenv
from images import upload_img

load_dotenv()


app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

######## HELPERS
# regex expression found at https://stackoverflow.com/questions/839994/extracting-a-url-in-python 
# from user: https://stackoverflow.com/users/8794221/allan
# Accessed: Dec 2022
#tested to confirm that it handles cases in which no protocol is specified for a given url
# input: any string
# Output: 2-element list where 1st element is a bool noting wether a url was found(for logic control in other methods)
#                              2nd element is the URL if one is found, ''(empty string) if not
def url_check(text:str):
    regex=r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"
    url = re.findall(regex, text)
    if url:
        return [True, url]
    return [False, url]

#Set a rating threshold
def rating(cb_score):    
    if cb_score > 0.49:
        return "Truthful"
    elif cb_score > 0.35:
        return "Questionable"
    else:
        return "False and/or Misleading"

#Webhook for receiving text submissions
@app.post("/webhook", status_code = status.HTTP_202_ACCEPTED)
async def recieve_message(
    request: Request, From: str = Form(...), Body: str = Form(...), NumMedia: int = Form(...), 
    ProfileName: str = Form(...) 
):
    ##parse request data
    #if its text: do this
    #else its a picture so do this instead
    
    payload = await request.form()
    response = MessagingResponse()

    #validate that submission is a URL
    is_url = url_check(Body)

    #check to see how many URL's we found in the submission
    # if more than 1 URL is found, let the user we can only handle one at a time
    if len(is_url) > 2:
        msg = response.message(f"""Hello {ProfileName}! Thank you for your submission
                                /nWe see that you submitted multiple URL's, please only submit one at a time. Thank you!""")
        return Response(content=str(response), media_type='application/xml')
    
    if is_url[0]: #if we find a url in the submission string
        url = str(is_url[1][0])
        res_to_user = the_magic(url) #submit that url to both API's
        time.sleep(2)
        cb_score = float(res_to_user[2]) 
        confidence_level = rating(cb_score) #use our confidence score to return a meaningful statement on the contents veracity to the user

        msg = response.message(f"""Hi {ProfileName}, Thank you for your submission! 
                \nHere are the results for the URL you submitted:\n\n{res_to_user[0]} \n{res_to_user[1]} 
                \nThis means the article you submitted is {confidence_level}""")
        return Response(content=str(response), media_type='application/xml')
    # the below block handles the case in which no url is found, this means the submission is raw text. Google only handles URL's so we only submit to claimbusters.
    else:
        res_to_user = submit_to_claimbust(Body)
        cb_score = res_to_user['results'][0]['score']
        confidence_level = rating(cb_score)
        msg = response.message(f"""Hi {ProfileName}, the query you submitted was {res_to_user['results'][0]['text']}
                                \n'According to Claim Busters, your query had a score of {cb_score}
                                \nThis means the query you submitted is {confidence_level} """)
        return Response(content=str(response), media_type='application/xml')



#Twilio allows you to configure a back-up webhook in case of the 1st one failing
#While not exactly an elegant solution, when an image is submitted our text webhook rejects it as an unprocessable entity
# this triggers a ping to the back-up webhook which is configured to process MMS, we upload the image to cloudinary
# which handles duplicate checking and stores the image in thier cloud storage
# For the time being we simply acknowledge the submission, as an extension an API(such as sensityAI) would be called to help form a useful response for the user
@app.post("/webhook2", status_code = status.HTTP_202_ACCEPTED)
async def handle_incoming_mms(request: Request, ProfileName: str = Form(...), MediaUrl0: str = Form(...)):
    # Get the contents of the MMS message
    response = MessagingResponse()
    payload = await request.form()
    # media_url = payload['MediaUrl0']
    # text = payload['Body']
    # profile_name = payload['ProfileName']

    img = upload_img(MediaUrl0)
    msg = response.message(f"Hi {ProfileName}, your MMS message was received! You can find it at: {img['url']} \n Unfortunately we can't process image submissions at this time, we will use your image to test our ability to fact-check images. Thank you!")
    # Save the MMS message or forward it to the appropriate recipient
    # ...

    # Return a response to Twilio to acknowledge receipt of the message
    return Response(content=str(response), media_type='application/xml')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)

# command to run app in reload mode(have server listen for all changes instead of having to manually run app everytime): py -m uvicorn main:app --reload
# requirements.txt contains all installed packages
# to install all packages listed run: pip install -r requirements.txt
# NEEDED ADDITIONS to requirements.txt
# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# pip install -U pip setuptools wheel
# python -m spacy download en_core_web_sm
#pip install mysql-connector
#pip install mysql-connector-python
#pip install python-multipart

