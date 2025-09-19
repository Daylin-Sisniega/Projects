document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("moviefile"); 
    const yearFilter = document.getElementById("movie-year"); 
    const directorFilter = document.getElementById("movie-director");
    const orderFilter = document.getElementById("movie-order"); 
    const searchInput = document.getElementById("movie-search"); 
    const movieContainer = document.getElementById("movie-posters"); 

    let movies = [];

    class Movie {
        constructor(title, director, releaseDate, imdbRating, posterUrl) {
            this.title = title;
            this.director = director;
            this.releaseDate = releaseDate;
            this.imdbRating = imdbRating;
            this.posterUrl = posterUrl;
        }
    }

    fileInput.addEventListener("change", function (event) {
        let file = event.target.files[0];
        let reader = new FileReader();

        reader.onload = function (event) {
            let contents = event.target.result;
            let json = JSON.parse(contents);  
            movies = []; 
            for (let i = 0; i < json.movies.length; i++) {
                let movie = json.movies[i];
                let movieObj = new Movie(movie.title, movie.director, movie.releaseDate, movie.imdbRating, movie.posterUrl);
                movies.push(movieObj); 
            }

            displayMovies(movies);
            populateYearFilter();
            populateDirectorFilter();

        };

        reader.onerror = function (event) {
            console.error("File could not be read! Code " + event.target.error.code); 
        };

        reader.readAsText(file); 
    });

    function displayMovies(movies) {
        const movieContainer = document.getElementById("movie-posters");

        movieContainer.innerHTML = ""; 

        movies.forEach(movie => {
            const movieCard = document.createElement("div");
            movieCard.classList.add("movie");

            const posterPath = `images/${movie.posterUrl}`;

            movieCard.innerHTML = `
               <img src="${posterPath}" alt="${movie.title}">
               <div class= "movie-info">
                <div class="movie-title">${movie.title}</div>
                <div class="movie-director">${movie.director}</div>
                <div class="movie-year-rating">${new Date(movie.releaseDate).getFullYear()} | IMDb: ${movie.imdbRating}</div>
                </div>
            `;
            movieContainer.appendChild(movieCard); 
         });
    }

    function populateYearFilter() {
        const uniqueYears = new Set();
        movies.forEach(movie => {
            const year = new Date(movie.releaseDate).getFullYear();
            uniqueYears.add(year);
        });
                            
        const sortedYears = Array.from(uniqueYears).sort((a, b) => b - a);
        yearFilter.innerHTML = '';
        const allYearsOption = document.createElement("option");
        allYearsOption.value = "";
        allYearsOption.textContent = "All Years";
        yearFilter.appendChild(allYearsOption);

        sortedYears.forEach(year => {
            let option = document.createElement("option");
            option.value = year;
            option.textContent = year;
            yearFilter.appendChild(option);
        });
    }

    yearFilter.addEventListener("change", function () {
        const selectedYear = yearFilter.value;
        const selectedDirector = directorFilter.value;
        searchInput.value = "";
        let filteredMovies = movies;
    
        if (selectedYear !== "") {
            filteredMovies = filteredMovies.filter(movie => new Date(movie.releaseDate).getFullYear() === parseInt(selectedYear));
        }
    
        if (selectedDirector !== "") {
            filteredMovies = filteredMovies.filter(movie => movie.director === selectedDirector);
        }
    
        displayMovies(filteredMovies); 
    });

    function populateDirectorFilter() {
        const uniqueDirectors = new Set();

        movies.forEach(movie => {
            uniqueDirectors.add(movie.director);
        });

        const sortedDirectors = Array.from(uniqueDirectors).sort();

        directorFilter.innerHTML = '';

        const allDirectorsOption = document.createElement("option");
        allDirectorsOption.value = "";
        allDirectorsOption.textContent = "All Directors";
        directorFilter.appendChild(allDirectorsOption);

        sortedDirectors.forEach(director => {
            let option = document.createElement("option");
            option.value = director;
            option.textContent = director;
            directorFilter.appendChild(option);
        });
    }
    directorFilter.addEventListener("change", function () {
        const selectedDirector = directorFilter.value;
        const selectedYear = yearFilter.value;
        searchInput.value = "";
        let filteredMovies = movies;
    
        if (selectedYear !== "") {
            filteredMovies = filteredMovies.filter(movie => new Date(movie.releaseDate).getFullYear() === parseInt(selectedYear));
        }
    
        if (selectedDirector !== "") {
            filteredMovies = filteredMovies.filter(movie => movie.director === selectedDirector);
        }
    
        displayMovies(filteredMovies); 
    });

    orderFilter.addEventListener("change", function () {
        const selectedOrder = orderFilter.value; 
        const selectedYear = yearFilter.value;
        const selectedDirector = directorFilter.value;  
        searchInput.value = "";

        let filteredMovies = [...movies]; 
        if (selectedYear !== "") {
            filteredMovies = filteredMovies.filter(movie => new Date(movie.releaseDate).getFullYear() === parseInt(selectedYear));
        }
        if (selectedDirector !== "") {
            filteredMovies = filteredMovies.filter(movie => movie.director === selectedDirector);
        }
        if (selectedOrder === "Ascending") {
            filteredMovies.sort((a, b) => new Date(a.releaseDate) - new Date(b.releaseDate));

        } else if (selectedOrder === "Descending") {
            filteredMovies.sort((a, b) => new Date(b.releaseDate) - new Date(a.releaseDate));
        }

        displayMovies(filteredMovies); 
    });

    function searchPart(searchQuery) {
        let filteredMovies = movies.filter(movie => {
            const movieYear = new Date(movie.releaseDate).getFullYear();
            const selectedYear = yearFilter.value; 
            if (selectedYear !== "" && movieYear !== parseInt(selectedYear)) {
                return false; 
            }
            return true; 
        });
    
        filteredMovies = filteredMovies.filter(movie =>
            movie.title && typeof movie.title === "string" &&
            movie.title.toLowerCase().includes(searchQuery.toLowerCase())
        );
        displayMovies(filteredMovies);
    }
    
    searchInput.addEventListener("input", function () {
        const searchQuery = searchInput.value.trim();  
        yearFilter.value = ""; 
        directorFilter.value = ""; 
    
        orderFilter.value = "Ascending"; 
        if (searchQuery === "") {
            let filteredMovies = movies.filter(movie => {
                const movieYear = new Date(movie.releaseDate).getFullYear();
                return yearFilter.value === "" || movieYear === parseInt(yearFilter.value);
            });
            displayMovies(filteredMovies);
        } else {
            searchPart(searchQuery);
        }
    });
    
});
