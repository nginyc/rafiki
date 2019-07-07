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
