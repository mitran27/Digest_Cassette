Designed and Developed an AI Powered Application which convert’s given text Image to a summarized podcast


the system design for the entire application is given below


![design](https://user-images.githubusercontent.com/62206653/145774647-0869f735-7882-4bb5-8d48-75b83a9bc98a.png)

<h1> Segmentation </h1>

The input image is segmented to find the text locations using FPN and differentiable binarization network

![seg](https://user-images.githubusercontent.com/62206653/145774446-2987a36c-270b-4cc9-aea1-cbb75a0a0ee8.jpeg)


<h1> Recognition </h1>

The text locations are croped and recognized
![ocr1](https://user-images.githubusercontent.com/62206653/145774520-b6d23888-c712-4aea-af4a-19c087846388.jpeg)

The recognized text is sorted and sent to summarization model

<h1> Summarization </h1>

the text is summarized using transformers

Sample Result-1:

<b>Source:</b> 
summstart ride hailing startup uber s main rival in 
southeast asia, grab has announced plans to open a research 
and development centre in bengaluru. the startup is looking 
to hire around 200 engineers in india to focus on developing 
its payments service grabpay. however, grab s engineering 
vp arul kumaravel said the company has no plans of 
expanding its on demand cab services to india. summend 

<b>Model output </b>:
summstart uber to open research centre in 
bengaluru summend

<b>Ground truth</b> :
summstart uber rival grab to open research 
centre in bengaluru summend

Sample Result-2:
<b>Source</b>:
summstart wrestlers geeta and babita phogat along 
with their father mahavir attended aamir khan s 52nd 
birthday celebrations at his mumbai residence. dangal is 
based on mahavir and his daughters. the film s director 
nitesh tiwari and actors aparshakti khurrana, fatima shaikh 
and sakshi tanwar were also spotted at the party. shah rukh 
khan and jackie shroff were among the other guests. 
summend

<b>Model output</b>: 
summstart geeta babita phogat attend 
birthday party s laga aamir summend

<b>Ground truth</b>:
summstart geeta, babita, phogat family attend 
aamir s birthday party summend

<h1> Aligner </h1>


the summarized text is converted to spectrogram using sequence to sequence model with location awareness



https://user-images.githubusercontent.com/62206653/145774055-396b547e-58c9-40cc-907c-6776443038b8.mp4

output without location awarness be like

![wolocationawr](https://user-images.githubusercontent.com/62206653/145774243-59e726c3-b6ae-4873-805c-c8cc2ffcfaca.png)

<h1> Vocoder </h1>

The spectrgroam generates the audio using Generative adversial network


https://user-images.githubusercontent.com/62206653/145774069-dc0c567a-9a74-429a-9975-da621f0bc016.mp4

The final end application built using angular and fastapi is displayed below


![WhatsApp Image 2021-11-27 at 1 25 28 AM](https://user-images.githubusercontent.com/62206653/145774340-3c9e7b00-94a5-4d7b-9e98-3ecc8c5d21ec.jpeg)

