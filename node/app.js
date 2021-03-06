const express = require('express');
const { recognize_number } = require('../pkg/recognize_number.js');

const app = express();
const port = 8080;
app.use('/public', express.static(__dirname + "/public"));
app.use(express.urlencoded({ extended: false }));

app.get('/', (req, res) => res.sendFile(__dirname + "/public/" + "index.html"));;

app.post('/recognize', function (req, res) {
	//console.log(`POST: ${req.body.data}`);
	//console.log(`imagedata.length: ${imagedata.length}`);
	let recog = recognize_number(req.body.data);
	console.log(`recognize result: ${recog}`);
	res.send(recog);
});

app.listen(port, () => console.log(`Listening at http://localhost:${port}`))

