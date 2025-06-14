.PHONY: setup install-precursor install-receptor install-parkour dev build clean

# Setup the entire project
setup: install-precursor install-receptor install-parkour

# Install Python dependencies for d3-precursor
install-precursor:
	cd d3-precursor && pip install -r requirements.txt

# Install Node.js dependencies for d3-receptor
install-receptor:
	cd d3-receptor && npm install

# Install Node.js dependencies for d3-parkour
install-parkour:
	cd d3-parkour && npm install

# Start development servers
dev:
	npm run dev

# Build for production
build:
	npm run build

# Docker operations
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Clean build artifacts
clean:
	rm -rf d3-precursor/dist d3-precursor/build d3-precursor/**/__pycache__
	rm -rf d3-receptor/dist d3-receptor/out
	rm -rf d3-parkour/.next d3-parkour/out

# Add a model to Ollama
add-model:
	@echo "Adding model to Ollama..."
	@cd scripts && ./add-model.sh

# Train the D3 Neuro model
train-model:
	@echo "Training D3 Neuro model..."
	@cd d3-precursor && python -m scripts.train_model 