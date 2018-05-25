/**
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http:// www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==============================================================================
*/

// Tiny TFJS train / predict example.
var model = "None";
var prepjson = "None";
var charRev = "None";
var charMap = "None";
var x = "None";
var x1_len = "None";
var x2_len = "None";

async function getModel() {
  // Create a simple model.
  console.log("getting model");
  model = await tf.loadModel("model_11/model.json");
  console.log("got model");
}

function loadJSON(callback) {   
   var xobj = new XMLHttpRequest();
   xobj.overrideMimeType("application/json");
   xobj.open('GET', 'model_11/prep.json', false);
   xobj.onreadystatechange = function () {
         if (xobj.readyState == 4 && xobj.status == "200") {
           callback(xobj.responseText);
         }
   };
   xobj.send(null);  
}

async function getJSON(){
    console.log("getting json");
    loadJSON(function(response, out) {
        prepjson = JSON.parse(response);
        console.log(prepjson);
     });
    charRev = prepjson["charRev"];
    charMap = prepjson["charMap"];
    x1_len = prepjson["x1_len"];
    x2_len = prepjson["x2_len"];
}

async function init(){
  document.getElementById('beginning_warn').innerText = "Loading model...";
  console.log("calling init");
  console.log("calling getModel from init");
  await getModel();
  console.log("calling getJSON from init");
  await getJSON();
  x = inner_to_x2_x1(charMap, x2_len, x1_len);
  for (iter = 0; iter < 100; iter++){
      "Predicting character "+iter;
      update_text(model, x, charRev, '', '');
  }
}

function inner_to_x2_x1(charMap, lenx2, lenx1){
  
  console.log(charMap);
  var name = document.getElementById('rec_name').innerText.toLowerCase().trim();
  var txt = "name:\n"+name+"\n\ningredients:\n\n";
  console.log("I found the following txt to start:  "+txt);
  document.getElementById('micro_out_div').innerText = txt;
  var mapped = []
  for (var i = 0; i < lenx2; i++){
    mapped.push(0);
  }
  start = lenx2 - txt.length;
  for (var i = start; i < txt.length; i++){
    console.log(i);
    console.log(txt.charAt(i));
    mapped[i] = charMap[txt.charAt(i)];
  }
  const x2 = tf.tensor2d([mapped]);
  const x1 = tf.tensor2d([mapped.slice(lenx2-lenx1, lenx2)]);
  x1.print();
  x2.print();
  console.log(x1.shape);
  console.log(x2.shape);
  return [x1, x2];
}

function sample(p, T){
  //console.log(p);
  //sample from an array with a given temperature
  for (var i = 0; i < p.length; i++){
    p[i] = Math.log(p[i])/T;
  }
  //console.log(p);
  var sum = 0;
  for (var i = 0; i < p.length; i++){
    p[i] = Math.pow(Math.E, p[i]);
    sum = sum + p[i];
  }
  //console.log(p);
  var samp_sum = 0;
  for (var i = 0; i < p.length; i++){
    p[i] = p[i]/sum + samp_sum;
    samp_sum = p[i];
  }
  //console.log(p);
  r = Math.random();
  for (ind = 0; ind < p.length; ind++){
    var threshold = p[ind];
    if (r < threshold){
      return ind;
    }
  }
  return ind;
}

function update_text(model, x, charRev, text_elem, prob_elem){
  var p = model.predict(x)[0].dataSync();
  ind = sample(p, 1);
  next_char = charRev[ind];
  x[0] = x[0].slice([0, 1],[1, x[0].shape[1]-1]).concat(tf.tensor2d([[ind]]), 1);
  x[1] = x[1].slice([0, 1],[1, x[1].shape[1]-1]).concat(tf.tensor2d([[ind]]), 1);
  console.log(x[0]);
  console.log(x[1]);
  document.getElementById('micro_out_div').innerText += next_char;
}


//init();
