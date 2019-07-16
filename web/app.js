/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

const express = require('express');	
const ejs = require('ejs');	
const app = express();	
const path = require('path');	

 const port = 3001;	

 // Set rendering enginer as EJS	
// This setup allows .html files to be rendered as EJS	
// Also sets a custom delimiter for EJS	
app.set('view engine', 'html');	
app.engine('html', ejs.renderFile); 	
app.set('view options', { delimiter: '?' }); 	
app.set('views', path.join(__dirname, 'build'));	

 app.use('/', express.static(path.join(__dirname, 'build')));	
app.get('/*', (req, res) => {	
  res.render('index', {	
    'REACT_APP_API_POINT_HOST': process.env.RAFIKI_ADDR,	
    'REACT_APP_API_POINT_PORT': process.env.ADMIN_EXT_PORT	
  });	
});	

 app.listen(port, () => {	
  console.log(`Rafiki Web Admin listening on port ${port}!`)	
});  	
