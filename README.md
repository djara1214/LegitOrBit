## **Legit or Bit** 	

    is it legit or a bit?

Legit or Bit is a fact checking service bot that lives in Whats App. This bot is designed for those who have Meta subsided internet plans. These plans offer extra gibibytes or unlimited data for Meta products **ONLY**. It is unreasonable to expect people without easy and reliable internet access to be meticulous and thorough in verifying the veracity of the information they receive. So this tool offers those who have these plans a way to fact check websites and claims that may be shared to them or they may have themselves; without the risk of overages and charges on their phone plans.
We are powered by the Claim Busters Fact Checking API created by The University of Texas at Arlington. Which scores all user submissions. We also provide similar articles and information for article submissions which is done by contacting the Google Fact Explorer API.  

This Project was created by Danny Jaramillo and Omar Stevens as their Senior Capstone Project for The City College of New York

**GO BEAVERS**

## Folder Contents 📁
 - `Db.py` - Connection to CloudSql, and all operations needed for database operations: such as pulling in data, modifications, and saving.  `Created by Omar Stevens`
 - `TextProcessing.py`- Contains all fact checking API calls, data-frame operations for saving query info, and user formatted returns. Used to run and check API responces	*Can be used to test API outputs*  `Created by Danny Jaramillo`
- `main.py` - Connections to web hooks, some user formatted response. Run to test with tunnel connected.   `Created by Omar Stevens`
- `requirments.txt` All dependencies needed for the environment
	- Additional requirements are located in `main.py`
- **Subfiles need but not uploaded**
	- **SSL**  - Keys needed for cloudSQLg
  - `.env` - Secret keys...
  - `images.py` - Connection to Cloudinary, and saving image submissions.

## How to 🤷‍♀️
#### How to use Legit or Bit as a user:
Send us a message through a device with Whats App installed.

📲 +13479054969

We accept any non-pay walled news article URL or a text submission:
#### Some examples:

     ➡ Was Tom Sawyer a real person

     ➡Is Messi Taller than Ronaldo

     ➡City College of New York is the best school for computer science

     ➡https://nonpaywalled_newsarticle.com


#### How to use Legit or Bit as a developer: 💻
*Note: For the following Webhooks must be set from Ngrok inside the Twilio console. They must have the same ip-address from Ngrok.*

*Inside the Twilio console, webhook 1: is for text/url processing* 	`/webhook`

*webhook 2: To accept image submissions*		`/webhook2`

-----------------------------------------------------------------------------------------------------------------

Download the Legit or bit Folder located in this repo.

Install any requirements with the `requirements.txt` file.
You can use your package manager to do so, for example if you have pip you can run the command `pip install -r requirements.txt`

Additional set-up: These are also listed in `main.py`

	 	`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib`
	
	 	`pip install -U pip setuptools wheel`
	
	 	`python -m spacy download en_core_web_sm`
		
		`pip install mysql-connector`
		
		`pip install mysql-connector-python`
		
		`pip install python-multipart`

Install Ngrok at https://ngrok.com/

Download their provided `Ngrok.exe`

Run it and make sure your account is verified ✅

Open Ngrok.exe

type in the command `ngrok.exe http 80` to get it running

Copy the Forwarding address 

* It should look something like this: 'https://some_numbers_yada_yada.ngrok.io'

Login to Twilio page and go to: 
	
* Messaging in the side bar ➡ Senders ➡ WhatsApp Senders ➡ Scroll down to our WhatsApp number and click it. 🖱

Copy and paste the forwarding address to the ***Webhook URL for incoming messages*** and make sure to leave the */webhook* without any space in between. 

Finally scroll down and hit the blue button that says *Update WhatsApp Sender*
  
Go back to your favorite IDE 💻 and run `py -m uvicorn main:app --reload`
  
The Ngrok tunnel will be connected to your console and you'll be able to see any and all traffic 🚦 to our code base.
	

*If you run into any errors, try to submit the same query again. This usually does the trick*

	🏁🏁 Now we are ready to rock and roll 🏁🏁

## Sample Images

<img src="https://user-images.githubusercontent.com/44910053/207897300-d785c3bd-6133-4753-beb6-b127c12bd59e.png" width=40% height=40%>   <img src="https://user-images.githubusercontent.com/44910053/207901492-76fa2009-43e9-4ce4-a2fa-cb752ea5a4e6.jpg" width=40% height=40%>

## Notes:

This application is still in testing and production, if you the user or developer messages the number provided above without having the Ngrok tunnel set up, you will recive no output. The tunnel currently is the only way we have connection to our code base.

We currently can not process images, for the future we want to have that service added to our project. 

We do accept images, but it will be used for future developments in fact checking them.

Our plan:

Information to send the user about the image:
		
		Meta Data
		
		If the image was altered
		
		Original source (done via reverse image search)
		
We also would like to find an additional service to provide professionaly facted checked articles. Since the Google Fact Check Explorer is very limted.
		
		This can be in the form of a API
		
		Or Simply a large dataset we can match similar queries and pull out of.
		
If you would like to add to out project, shoot us a message. We would love colaborators and fresh new ideas added to this service.
So we may all together help those with limited internet access!!
	

 
