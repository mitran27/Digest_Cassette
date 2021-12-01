import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { FormsModule ,ReactiveFormsModule}   from '@angular/forms';



import { HttpClientModule } from '@angular/common/http';
import { ImageComponent } from './components/image/image.component';
import { WikiComponent } from './components/wiki/wiki.component';

@NgModule({
  declarations: [
    AppComponent,
    ImageComponent,
    WikiComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
