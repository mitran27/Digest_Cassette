import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ImageServiceService {

  constructor(private http: HttpClient) {}


  public uploadImage(image: File): any {
    const formData = new FormData();

    formData.append('image', image);

    return this.http.post('', formData);
  }
}
