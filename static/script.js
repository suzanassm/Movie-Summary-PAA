async function buscar() {
    const query = document.getElementById("query").value;
    const metodo = document.getElementById("metodos").value;
    let response;

    if (metodo != "tfidf") {
        response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query,
                method: metodo,
            })
        });

        const data = await response.json();
        console.log(data);

        document.getElementById("resultado").innerHTML = `
            <h2>Resposta:</h2>
            <p>${data.answer}</p>

            <h3>Filmes encontrados:</h3>
            <ul>
                ${data.movies_found.map(movie => `<li>${movie}</li>`).join("")}
            </ul>
        `;

    }
    else {
        response = await fetch("/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query,
                method: metodo,
            })
        });

        const data = await response.json();
        console.log(data);

        document.getElementById("resultado").innerHTML = `
        
            <h2>Filmes encontrados:</h2>
            <ul>
                ${data.results.map(movie => `<li>${movie}</li>`).join("")}
            </ul>
        `;

    }
}