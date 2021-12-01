import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {ImageComponent} from "./components/image/image.component"
import {WikiComponent} from "./components/wiki/wiki.component"

const routes: Routes = [
  { path: '', component: ImageComponent },
  { path: 'wiki', component: WikiComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
