// Mobile navigation toggle
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
        
        // Close menu when clicking on a link
        document.querySelectorAll('.nav-menu a').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Navbar background on scroll
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }
    
    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.overview-card, .module-card, .pattern-card, .step, .api-card').forEach(el => {
        observer.observe(el);
    });
    
    // Copy code functionality
    document.querySelectorAll('pre code').forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        const pre = block.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
    
    // Module card interactions
    document.querySelectorAll('.module-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Architecture diagram interactions
    document.querySelectorAll('.arch-component').forEach(component => {
        component.addEventListener('click', function() {
            const moduleName = this.textContent.split('\n')[0].toLowerCase();
            const moduleSection = document.querySelector(`#${moduleName}`);
            if (moduleSection) {
                moduleSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Search functionality (if search input exists)
    const searchInput = document.querySelector('#search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const searchableElements = document.querySelectorAll('.module-card, .api-card, .pattern-card');
            
            searchableElements.forEach(element => {
                const text = element.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    element.style.display = 'block';
                    element.style.opacity = '1';
                } else {
                    element.style.opacity = '0.3';
                }
            });
            
            if (searchTerm === '') {
                searchableElements.forEach(element => {
                    element.style.opacity = '1';
                });
            }
        });
    }
    
    // Theme toggle (if theme toggle exists)
    const themeToggle = document.querySelector('#theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
        });
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
        }
    }
    
    // Performance metrics animation
    const performanceMetrics = document.querySelectorAll('.metric-value');
    performanceMetrics.forEach(metric => {
        const finalValue = parseFloat(metric.textContent);
        let currentValue = 0;
        const increment = finalValue / 100;
        
        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= finalValue) {
                currentValue = finalValue;
                clearInterval(timer);
            }
            metric.textContent = currentValue.toFixed(1) + (metric.dataset.suffix || '');
        }, 20);
    });
    
    // Lazy loading for images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Tooltip functionality
    document.querySelectorAll('[data-tooltip]').forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.dataset.tooltip;
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
        });
        
        element.addEventListener('mouseleave', function() {
            const tooltip = document.querySelector('.tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
    
    // Progress bar for page scroll
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset;
            const docHeight = document.body.offsetHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            progressBar.style.width = scrollPercent + '%';
        });
    }
    
    // Module status indicators
    const moduleStatuses = {
        'orchestrator': 'active',
        'pretoria': 'active',
        'mzekezeke': 'active',
        'zengeza': 'active',
        'nicotine': 'active',
        'diadochi': 'active'
    };
    
    Object.keys(moduleStatuses).forEach(module => {
        const statusElement = document.querySelector(`#${module}-status`);
        if (statusElement) {
            statusElement.className = `status ${moduleStatuses[module]}`;
            statusElement.textContent = moduleStatuses[module].toUpperCase();
        }
    });
    
    // Interactive architecture diagram
    const archComponents = document.querySelectorAll('.arch-component');
    archComponents.forEach(component => {
        component.addEventListener('mouseenter', function() {
            // Highlight connected components
            const componentType = this.classList[1]; // Get the module type class
            document.querySelectorAll(`.${componentType}`).forEach(el => {
                el.style.boxShadow = '0 0 20px rgba(99, 102, 241, 0.5)';
            });
        });
        
        component.addEventListener('mouseleave', function() {
            // Remove highlights
            archComponents.forEach(el => {
                el.style.boxShadow = '';
            });
        });
    });
    
    // Code syntax highlighting enhancement
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
    
    // Analytics tracking (if needed)
    function trackEvent(category, action, label) {
        if (typeof gtag !== 'undefined') {
            gtag('event', action, {
                event_category: category,
                event_label: label
            });
        }
    }
    
    // Track module card clicks
    document.querySelectorAll('.module-link').forEach(link => {
        link.addEventListener('click', function() {
            const moduleName = this.closest('.module-card').querySelector('h3').textContent;
            trackEvent('Navigation', 'Module Click', moduleName);
        });
    });
    
    // Track API documentation clicks
    document.querySelectorAll('.api-list a').forEach(link => {
        link.addEventListener('click', function() {
            trackEvent('Documentation', 'API Click', this.textContent);
        });
    });
    
    console.log('🎆 Spectacular Documentation Site Loaded');
}); 