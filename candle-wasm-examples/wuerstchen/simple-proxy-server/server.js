const express = require('express');
const request = require('request');
const app = express();

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.get('/fetch-resource', (req, res) => {
    const url = req.query.url;

    console.log('fetch resource to ' + url);

    let newReq = request.get(url);
    newReq.pipe(res);

    //let url = "http://localhost:80/file.txt";
    //request.get(url).pipe(res);
    //request(url).pipe(res);
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});