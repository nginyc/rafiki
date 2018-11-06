const express = require('express');
const path = require('path');
const app = express();

const port = 8080;

app.use('/node_modules', express.static('./node_modules'));
app.use('/dist', express.static('./dist'));

app.get('/*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
  console.log(`Rafiki Admin Web listening on port ${port}!`)
});