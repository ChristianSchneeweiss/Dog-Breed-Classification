document.querySelector("#Predict").addEventListener("click", () => {
    postData("http://localhost:5000/predict", data)
});

inputElement = document.querySelector("#file");
inputElement.onchange = function (event) {
    data = new FormData();
    const file = inputElement.files[0];
    data.append("file", file);
    const preview = document.querySelector("#img");
    const reader  = new FileReader();

   reader.onloadend = function () {
       preview.src = reader.result;
   };

   if (file) {
       reader.readAsDataURL(file); //reads the data as a URL
   } else {
       preview.src = "";
   }

    document.querySelector("#card").classList.remove("d-none");
   document.querySelector(".btn.btn-primary").removeAttribute("disabled")
};
let data = {};

function postData(url = '', data) {
    // Default options are marked with *
    return fetch(url, {
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        mode: 'cors', // no-cors, cors, *same-origin
        cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        credentials: 'same-origin', // include, *same-origin, omit
        // headers: {
        //     'Content-Type': 'application/json',
        //     // 'Content-Type': 'application/x-www-form-urlencoded',
        // },
        redirect: 'follow', // manual, *follow, error
        referrer: 'no-referrer', // no-referrer, *client
        body: data // body data type must match "Content-Type" header
    })
        .then(async response => {
            const prediction = (await response.json()).prediction;
            document.querySelector("#prediction-text").innerHTML = prediction;
        });
}