const express = require('express');
const path = require('path');
const app = express();

const port = 3001;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')));
app.use('/dist', express.static(path.join(__dirname, 'dist')));

app.get('/*', (req, res) => {
  res.render('index', {
    'ADMIN_HOST': process.env.RAFIKI_ADDR,
    'ADMIN_PORT': process.env.ADMIN_EXT_PORT
  });
});

app.listen(port, () => {
  console.log(`Rafiki Admin Web listening on port ${port}!`)
});