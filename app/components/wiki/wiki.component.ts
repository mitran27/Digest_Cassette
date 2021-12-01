import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { HttpClient ,HttpHeaders} from '@angular/common/http';
import { stringify } from 'querystring';

@Component({
  selector: 'app-wiki',
  templateUrl: './wiki.component.html',
  styleUrls: ['./wiki.component.css'],
})
export class WikiComponent implements OnInit {
  SERVER_URL = 'http://localhost:3000/upload';

  isDisplay = false;
  audioContent;
  title = '';
  lines;
  type = '';

  stage1 =false;
  stage2=false;

  constructor(private httpClient: HttpClient) {}
  reset(){
    this.stage1=false
    this.stage2=false
    this.isDisplay=true
  }
  onSubmit() {

    this.reset()
    const formData = new FormData();
    formData.append('type', this.type);
    formData.append('title', this.title);
    formData.append('lines', this.lines);

  

    this.httpClient.post<any>("http://127.0.0.1:8000/summarize",{text: this.title,option:1,length:this.lines}).subscribe(

          (res1) => {
            console.log(res1)        
            this.stage1=true
  
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
            sound.volume  =  0.00007;
            document.getElementById('result').appendChild(sound);

            
            this.stage2=true
            this.isDisplay=false
            console.log(this.audioContent)
    
            
            
  
  
  
  
          })


          } )



      }
     

  ngOnInit() {}
}
