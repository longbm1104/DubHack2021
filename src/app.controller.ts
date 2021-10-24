import { Controller, Get, Param } from '@nestjs/common';
import { AppService } from './app.service';
const nodePickle = require('node-pickle');

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get()
  getHello(): string {
    return this.appService.getHello();
  }

  @Get("image/:pathFile")
  predictImg(@Param("pathFile") file: string): string {
    let fullPath = "../input/fruits-360/Training/Cocos/" + file;
    module = nodePickle.load('svm_model.pkl').then(data => console.log(data));
    return file;
  }
}