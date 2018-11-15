const express = require('express');
const path = require('path');
const app = express();

const port = 8080;

app.set('view engine', 'ejs');
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')));
app.use('/dist', express.static(path.join(__dirname, 'dist')));

app.get('/*', (req, res) => {
  res.render('index', {
    'ADMIN_HOST': process.env.RAFIKI_IP_ADDRESS,
    'ADMIN_PORT': process.env.ADMIN_EXT_PORT
  });
});

app.listen(port, () => {
  console.log(`Rafiki Admin Web listening on port ${port}!`)
});