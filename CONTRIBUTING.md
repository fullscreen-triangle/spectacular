# Contributing to D3-Neuro

Thanks for considering contributing to D3-Neuro! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template to create a new issue
- Include detailed steps to reproduce the bug
- Provide specific examples (e.g., sample code that demonstrates the bug)
- Describe the expected behavior vs. the actual behavior

### Suggesting Enhancements

- Use the feature request template to suggest enhancements
- Provide a clear and detailed explanation of the feature/enhancement
- Explain why this enhancement would be useful to most D3-Neuro users
- Include examples of how this enhancement would be used

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Pull Request Guidelines

- Update documentation for any changed functionality
- Follow the existing code style
- Include tests that verify your changes
- Keep pull requests focused on a single feature/fix
- Link any related issues in the pull request description

## Development Process

### Setting Up the Development Environment

1. Clone your fork of the repository
2. Install the project dependencies as described in the README
3. Make sure all tests pass before making changes

### Project Structure

The project is divided into three main components:

- **d3-precursor**: Python backend (knowledge extraction, processing)
- **d3-receptor**: React frontend (LLM integration, visualization)
- **d3-parkour**: Next.js web application (deployment platform)

### Testing

- Run tests for the component you're working on:
  - Python tests: `cd d3-precursor && pytest`
  - React tests: `cd d3-receptor && npm test`
  - Next.js tests: `cd d3-parkour && npm test`
- Add tests for new functionality
- Ensure existing tests pass

### Coding Style

- Follow the existing code style
- For Python code:
  - Follow PEP 8 guidelines
  - Use type hints
  - Document functions and classes with docstrings
- For JavaScript/TypeScript code:
  - Follow the ESLint configuration
  - Use TypeScript interfaces for component props
  - Document components and functions with JSDoc comments

## License

By contributing to D3-Neuro, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project. 