import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { HttpClient ,HttpHeaders} from '@angular/common/http';

@Component({
  selector: 'app-image',
  templateUrl: './image.component.html',
  styleUrls: ['./image.component.css'],
})
export class ImageComponent implements OnInit {
  SERVER_URL = 'http://localhost:3000/upload';
  uploadForm: FormGroup;
  isDisplay = false;
  audioContent;
  type;
  image;
  fileToUpload: File | null = null;

  stage1=false
  stage2=false 
  stage3=false 

  

  


  constructor(
    private formBuilder: FormBuilder,
    private httpClient: HttpClient
  ) {}



  onFileChanged(event) {
    this.fileToUpload = event.target.files[0];
  }
  onType(value) {
    this.uploadForm.get('type').setValue(value);
  }

  reset(){
    this.stage1=false
    this.stage2=false
    this.stage3=false
    this.isDisplay=true
  }

  onSubmit() {
    
    this.reset()
    const formData = new FormData();

    formData.append('file', this.fileToUpload, this.fileToUpload.name);   
    console.log(formData);

    this.httpClient.post<any>("http://127.0.0.1:8000/ocr",formData).subscribe(
      (res) => {
        this.stage1=true


        let sourcetext=res["sentence"]
        console.log(sourcetext)
       //let sourcetext="summstart the enforcement directorate has summoned jammu and kashmir liberation front chief yasin malik and separatist leader syed ali shah geelani in different money laundering cases. one of the cases pertains to the income tax department s recovery of 10,000 at geelani s residence. similarly, in 2001 police had seized 100,000 from a yasin malik aide which was to be handed to him. summend";
       this.httpClient.post<any>("http://127.0.0.1:8000/summarize",{text:sourcetext,option:0,length:1}).subscribe(

        (res1) => {
          console.log(res1)
          this.stage2=true



          const httpOptions = {
            headers: new HttpHeaders({
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }),
            responseType: 'blob' as 'json'
          };         

             

          this.httpClient.post<any>("http://127.0.0.1:8000/podcast",{text:res1},httpOptions).subscribe(
        (res2) => {

          console.log(res2)   
          
          var url=URL.createObjectURL(res2)
          this.audioContent=url
          /*
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = 'test.wav';
          document.body.appendChild(a);
          a.click();*/

          document.getElementById('result').removeChild(document.getElementById('result').firstChild)

          var sound      = document.createElement('audio');
          sound.id       = 'audio-player';
          sound.controls = true;
          sound.src      = url;
          sound.volume   = 0.00001;
          document.getElementById('result').appendChild(sound);

          
          this.stage3=true
          this.isDisplay=false
          console.log(this.audioContent)

          
          




        })

        })


      },
      (err) => {
        console.log(err);
      }
    );


  
  }
  ngOnInit() {
    this.uploadForm = this.formBuilder.group({
      profile: [''],
      type: '',
    });
  }
}
