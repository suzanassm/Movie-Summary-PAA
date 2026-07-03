async function buscar() {
    const query = document.getElementById("query").value;
    const metodo = document.getElementById("metodos").value;

    const endpoint = metodo !== "tfidf" ? "/ask" : "/search";

    const carregamento = document.getElementById("carregamento");
    const resultado = document.getElementById("resultado");


    try {
        carregamento.style.display = "block";
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query,
                method: metodo
            })
        });

        const data = await response.json();
        console.log(data);

        let html = "";

        if (metodo !== "tfidf") {
            html = `
                <h2>Resposta:</h2>
                <p>${data.answer}</p>

                <h3>Filmes encontrados:</h3>
                <ul>
                    ${(data.movies_found || [])
                        .map(movie => `<div class="movie-card">${movie}</div>`)
                        .join("")}
                </ul>
            `;
        } else {
            html = `
                <h2>Filmes encontrados:</h2>
                <ul>
                        ${(data.results || [])
                            .map(movie => `<div class="movie-card">${movie}</div>`)
                            .join("")}
                </ul>
            `;
        }

        document.getElementById("resultado").innerHTML = html;

    } catch (error) {
        console.error("Erro na busca:", error);
    } finally {
        carregamento.style.display = "none"
    }
}