const lda = require('lda');
const brain = require('brain.js');
const train_data = require('./train_data1.json');

const title = 'cat lovers';
const desc = 'keep all the talking to cat';
const text = title.toLowerCase()+' '+desc.toLowerCase();

const documents = text.match(/[^\.!\?]+[\.!\?]+/g);
const result = lda(documents, 2, 5);
const term = result[0];

let extracted_term = title;

if(result.length>0){
    extracted_term = term[0].term;
}

const net = new brain.recurrent.LSTM();

net.train(train_data, {
    iterations: 10,
});

const output = net.run(extracted_term);

console.log('Category: ', output);
