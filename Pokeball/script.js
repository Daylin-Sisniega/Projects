// ChatGPT Conversation Links:
// 1.https://chatgpt.com/share/67f9d98f-bc38-8005-b0e9-4f556a5db20d
// 2.
// Add as many links as needed

const searchButton = document.getElementById("search-button");
const searchInput = document.getElementById("search-input");
const loadingSpinner = document.getElementById("loading");
const pokemonDetails = document.getElementById("pokemon-details");
const favouritesButton = document.getElementById("toggle-favourites");

const modal = document.getElementById("pokemon-modal");
const closeModal = document.getElementById("close-modal");

const modalName = document.getElementById("modal-name");
const modalId = document.getElementById("modal-id");
const modalImg = document.getElementById("modal-img");
const modalTypes = document.querySelector(".types-container");
const modalHeight = document.querySelector("#modal-height span");
const modalWeight = document.querySelector("#modal-weight span");
const modalAbilities = document.querySelector("#modal-abilities span");
const modalStats = document.getElementById("modal-stats");

const favouritesKey = "favourites";
let favourites = JSON.parse(localStorage.getItem(favouritesKey)) || [];


function showLoading() {
  loadingSpinner.classList.remove("hidden");
  pokemonDetails.innerHTML = "";
}

function hideLoading() {
  loadingSpinner.classList.add("hidden");
}

function showError(message) {
  pokemonDetails.innerHTML = `<p class="error">${message}</p>`;
}

async function fetchPokemonData(pokemon) {
  showLoading();
  try {
    const response = await fetch(`https://pokeapi.co/api/v2/pokemon/${pokemon.toLowerCase().trim()}`);
    if (!response.ok) {
      throw new Error("Pokémon not found");
    }
    const data = await response.json();
    hideLoading();
    displayMainCard(data);
  } catch (error) {
    hideLoading();
    showError("❌ Pokémon not found.");
  }
}
function formatId(id) {
  // Use padStart to format the ID for modal (e.g., #005)
  return id.toString().padStart(3, '0');
}

function displayMainCard(data) {
  const types = data.types.map((t) => t.type.name).join(", ");
  const abilities = data.abilities.map((a) => a.ability.name).join(", ");
  const img = data.sprites.other["official-artwork"].front_default || data.sprites.front_default;

  pokemonDetails.innerHTML = `
    <div class="pokemon-card">
      <img src="${img}" alt="${data.name}" class="card-img"/>
      <h2>${data.name}</h2>
      <p><strong>ID:</strong> ${data.id}</p> <!-- Just display the ID as it is here -->
      <p><strong>Types:</strong> ${types}</p>
      <p><strong>Abilities:</strong> ${abilities}</p>
      <button class="more-info-button">More Info</button>
      <button class="add-fav-button">⭐Add to Favourites</button>
    </div>
  `;

  document.querySelector(".more-info-button").addEventListener("click", () => {
    showModal(data);
  });

  document.querySelector(".add-fav-button").addEventListener("click", () => {
    addToFavourites(data);
  });
}

function showModal(data) {
  modalName.textContent = data.name;
  modalId.textContent = `#${formatId(data.id)}`; // Format the ID here for modal display
  modalImg.src = data.sprites.other["official-artwork"].front_default || data.sprites.front_default;

  modalHeight.innerHTML = `<span>${data.height / 10} m</span>`;
  modalWeight.innerHTML = ` <span>${data.weight / 10} kg</span>`;

  modalAbilities.textContent = data.abilities.map((a) => a.ability.name).join(", ");

  modalStats.innerHTML = data.stats
    .map((s) => `<div>${s.stat.name}: ${s.base_stat}</div>`)
    .join("");

  modal.classList.remove("hidden");
}


closeModal.addEventListener("click", () => {
  modal.classList.add("hidden");
});

searchButton.addEventListener("click", () => {
  const pokemon = searchInput.value;
  if (pokemon) {
    fetchPokemonData(pokemon);
  } else {
    showError("Please enter a Pokémon name or ID.");
  }
});

function addToFavourites(pokemon) {
  if (favourites.length < 6) {
    if (!favourites.some((fav) => fav.id === pokemon.id)) {
      favourites.push(pokemon);
      localStorage.setItem(favouritesKey, JSON.stringify(favourites));
      alert(`${pokemon.name} added to favourites!`);
    } else {
      alert("Already in favourites!!");
    }
  } else {
    alert("You can only favourite up to 6 Pokémon.");
  }
}

  function toggleFavouritesView() {
    if (favouritesButton.textContent === "View Favourites") {
      // Show favourites
      pokemonDetails.innerHTML = "";
      displayFavourites();
  
      
      searchButton.classList.add("hidden");
      searchInput.classList.add("hidden");
  
      // Change button to "Back to Search"
      favouritesButton.textContent = "Back to Search";
    } else {
      
      pokemonDetails.innerHTML = "";
      searchButton.classList.remove("hidden");
      searchInput.classList.remove("hidden");
  
      // Change button back to "View Favourites"
      favouritesButton.textContent = "View Favourites";
    }
  }

  function displayFavourites() {
    pokemonDetails.innerHTML = favourites
      .map((pokemon) => `
        <div class="pokemon-card">
          <img src="${pokemon.sprites.other['official-artwork'].front_default || pokemon.sprites.front_default}" alt="${pokemon.name}" class="card-img"/>
          <h2>${pokemon.name}</h2>
          <!-- Display ID without leading zeros -->
          <p><strong>ID:</strong> ${pokemon.id}</p> <!-- Just show the raw ID here -->
          <p><strong>Types:</strong> ${pokemon.types.map(t => t.type.name).join(", ")}</p>
          <p><strong>Abilities:</strong> ${pokemon.abilities.map(a => a.ability.name).join(", ")}</p>
          <button class="more-info-button">More Info</button>
        </div>
      `)
      .join("");
  
    document.querySelectorAll(".more-info-button").forEach((button, index) => {
      button.addEventListener("click", () => {
        showModal(favourites[index]);
      });
    });
  }
favouritesButton.addEventListener("click", toggleFavouritesView);
