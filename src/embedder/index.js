const {pipeline} = require('@xenova/transformers');

let embedder = null;

async function getEmbedder(){
    if(!embedder){
        console.log('Loading embedder model...');
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        console.log('Embedder model loaded.');
    }
    return embedder;
}

async function embed(text){
    const model = await getEmbedder();
    const output = await model(text, {pooling: 'mean', normalize: true});
    return Array.from(output.data);
}

module.exports = {embed};