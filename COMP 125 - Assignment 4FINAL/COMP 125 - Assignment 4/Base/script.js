// ChatGPT Conversation Links:
// 1.
// 2.
// Add as many links as needed

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("moviefile"); //DONT MODIFY
    const yearFilter = document.getElementById("movie-year"); //DONT MODIFY
    const directorFilter = document.getElementById("movie-director"); //DONT MODIFY
    const orderFilter = document.getElementById("movie-order"); //DONT MODIFY
    const searchInput = document.getElementById("movie-search"); //DONT MODIFY
    const movieContainer = document.getElementById("movie-posters"); //DONT MODIFY

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

    // File Change
    fileInput.addEventListener("change", function (event) {
        let file = event.target.files[0];
        let reader = new FileReader();

        reader.onload = function (event) {
            let contents = event.target.result;
            let json = JSON.parse(contents); // Parse the JSON data
            // console.log("JSON Loaded: ", json); // Check if the JSON is loaded correctly
            //console.log("Movies array: ", json.movies); 
            movies = []; // Reset the movies array

            // Loop through the movies array from the JSON and create movie objects
            for (let i = 0; i < json.movies.length; i++) {
                let movie = json.movies[i];
                //  console.log(`Movie ${i}:`, movie);  // Log the entire movie object to check for missing properties
                let movieObj = new Movie(movie.title, movie.director, movie.releaseDate, movie.imdbRating, movie.posterUrl);
                movies.push(movieObj); // Store in the global array
            }

            displayMovies(movies); // Call displayMovies to display the posters
            populateYearFilter();
            populateDirectorFilter();

        };

        reader.onerror = function (event) {
            console.error("File could not be read! Code " + event.target.error.code); // Handle read errors
        };

        reader.readAsText(file); // Read the file as text
    });

    // Function to Display Movies
    function displayMovies(movies) {
        // console.log("Displaying movies...");  // Check if this is logged
        const movieContainer = document.getElementById("movie-posters");
        //console.log(movieContainer);  // Check if the correct element is selected

        movieContainer.innerHTML = ""; // Reset the movie container
        // console.log(movieContainer.innerHTML);  // This will show the HTML content inside the container after resetting

        movies.forEach(movie => {
            const movieCard = document.createElement("div");
            // movieCard.className = "movie";
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
            movieContainer.appendChild(movieCard); // Append movie card
            //console.log(`Title: ${movie.title}, Director: ${movie.director}, Release Date: ${movie.releaseDate}, IMDb Rating: ${movie.imdbRating}`);
        });
    }

    // Function to populate the year filter dropdown with years from 2024 to 1995
    function populateYearFilter() {
        const uniqueYears = new Set();

        // this loop through movies and add the year to the set
        movies.forEach(movie => {
            const year = new Date(movie.releaseDate).getFullYear();
            uniqueYears.add(year);
        });

        // this will sort the years in descending order                             
        const sortedYears = Array.from(uniqueYears).sort((a, b) => b - a);

        // this is clearing the options on the dropdown
        yearFilter.innerHTML = '';

        // in this case the all is helping to display all the years 
        const allYearsOption = document.createElement("option");
        allYearsOption.value = "";
        allYearsOption.textContent = "All Years";
        yearFilter.appendChild(allYearsOption);

        // this shows the dropdown with the sorted years
        sortedYears.forEach(year => {
            let option = document.createElement("option");
            option.value = year;
            option.textContent = year;
            yearFilter.appendChild(option);
        });
    }

    // Filters
    yearFilter.addEventListener("change", function () {
        const selectedYear = yearFilter.value;
        const selectedDirector = directorFilter.value;
    
        // Clear search input when filters are changed
        searchInput.value = "";
    
        // Filter movies based on both the year and director
        let filteredMovies = movies;
    
        if (selectedYear !== "") {
            filteredMovies = filteredMovies.filter(movie => new Date(movie.releaseDate).getFullYear() === parseInt(selectedYear));
        }
    
        if (selectedDirector !== "") {
            filteredMovies = filteredMovies.filter(movie => movie.director === selectedDirector);
        }
    
        displayMovies(filteredMovies); // Display the filtered movies
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
    
        // Clear search input when filters are changed
        searchInput.value = "";
    
        // Filter movies based on both the director and year
        let filteredMovies = movies;
    
        if (selectedYear !== "") {
            filteredMovies = filteredMovies.filter(movie => new Date(movie.releaseDate).getFullYear() === parseInt(selectedYear));
        }
    
        if (selectedDirector !== "") {
            filteredMovies = filteredMovies.filter(movie => movie.director === selectedDirector);
        }
    
        displayMovies(filteredMovies); // Display the filtered movies
    });

    orderFilter.addEventListener("change", function () {
        const selectedOrder = orderFilter.value;  // Get the selected sorting order
        const selectedYear = yearFilter.value;  // Get the selected year
        const selectedDirector = directorFilter.value;  // Get the selected director

        // Clear search input when filters are changed
        searchInput.value = "";

        let filteredMovies = [...movies];  // Start with the full list of movies

        // Apply year filter if selected
        if (selectedYear !== "") {
            filteredMovies = filteredMovies.filter(movie => new Date(movie.releaseDate).getFullYear() === parseInt(selectedYear));
        }

        // Apply director filter if selected
        if (selectedDirector !== "") {
            filteredMovies = filteredMovies.filter(movie => movie.director === selectedDirector);
        }

        // Apply sorting based on selected order
        if (selectedOrder === "Ascending") {
            // Sort first by year in ascending order
            filteredMovies.sort((a, b) => new Date(a.releaseDate) - new Date(b.releaseDate));

        } else if (selectedOrder === "Descending") {
            // Sort first by year in descending order
            filteredMovies.sort((a, b) => new Date(b.releaseDate) - new Date(a.releaseDate));
        }

        displayMovies(filteredMovies); // Display the filtered and sorted movies
    });

    function searchPart(searchQuery) {
        // Filter the movies array using .filter() and .includes() (case insensitive)
        // First filter by year
        let filteredMovies = movies.filter(movie => {
            const movieYear = new Date(movie.releaseDate).getFullYear();
            const selectedYear = yearFilter.value; // Get the selected year from the dropdown
    
            // Check if the movie matches the selected year (if a year is selected)
            if (selectedYear !== "" && movieYear !== parseInt(selectedYear)) {
                return false; // Skip movies that don't match the selected year
            }
    
            return true; // Otherwise, keep the movie in the list
        });
    
        // Apply the search query filter
        filteredMovies = filteredMovies.filter(movie =>
            movie.title && typeof movie.title === "string" &&
            movie.title.toLowerCase().includes(searchQuery.toLowerCase())
        );
    
        // Display the filtered movies
        displayMovies(filteredMovies);
    }
    

    // Search
    searchInput.addEventListener("input", function () {
        // Get the search query from the input field
        const searchQuery = searchInput.value.trim();  // Trim to avoid any extra spaces
    
        // Clear the year filter when a search query is entered
        yearFilter.value = "";  // This will clear the year filter
        directorFilter.value = ""; // This will clear the director filter
    
        orderFilter.value = "Ascending";  // Reset to ascending order
        
        // Check if search query is empty or has a value
        if (searchQuery === "") {
            let filteredMovies = movies.filter(movie => {
                const movieYear = new Date(movie.releaseDate).getFullYear();
                return yearFilter.value === "" || movieYear === parseInt(yearFilter.value);
            });
            displayMovies(filteredMovies);
        } else {
            // Filter movies based on the search query and the current filters
            searchPart(searchQuery);
        }
    });
    
});
