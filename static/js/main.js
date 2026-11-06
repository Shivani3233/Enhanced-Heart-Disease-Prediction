// Main JavaScript for Heart Disease Prediction App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

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

    // Navbar scroll effect
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
    });

    // Form validation enhancement
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Focus on first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.focus();
                    firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
            form.classList.add('was-validated');
        });
    });

    // Input field animations
    const inputs = document.querySelectorAll('.form-control, .form-select');
    inputs.forEach(input => {
        // Add floating label effect
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });

        input.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });

        // Add input validation feedback
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            } else {
                this.classList.remove('is-valid');
                this.classList.add('is-invalid');
            }
        });
    });

    // BMI Calculator Helper
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');

    if (heightInput && weightInput && bmiInput) {
        function calculateBMI() {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = weight / (height * height);
                bmiInput.value = bmi.toFixed(1);
            }
        }

        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
    }

    // Age-based risk warnings
    const ageInput = document.getElementById('age');
    if (ageInput) {
        ageInput.addEventListener('input', function() {
            const age = parseInt(this.value);
            const warningDiv = document.getElementById('age-warning');
            
            if (age >= 65) {
                showAgeWarning('Age 65+ increases cardiovascular risk factors.');
            } else if (age >= 55) {
                showAgeWarning('Age 55+ is associated with increased heart disease risk.');
            } else if (warningDiv) {
                warningDiv.style.display = 'none';
            }
        });
    }

    function showAgeWarning(message) {
        let warningDiv = document.getElementById('age-warning');
        if (!warningDiv) {
            warningDiv = document.createElement('div');
            warningDiv.id = 'age-warning';
            warningDiv.className = 'alert alert-warning mt-2';
            ageInput.parentElement.appendChild(warningDiv);
        }
        warningDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${message}`;
        warningDiv.style.display = 'block';
    }

    // Real-time form validation feedback
    const predictForm = document.getElementById('predictionForm');
    if (predictForm) {
        // Add progress indicator
        addFormProgress();
        
        // Real-time validation
        const requiredFields = predictForm.querySelectorAll('[required]');
        requiredFields.forEach(field => {
            field.addEventListener('change', updateFormProgress);
            field.addEventListener('input', updateFormProgress);
        });
    }

    function addFormProgress() {
        const formHeader = document.querySelector('.text-center.mb-5');
        if (formHeader) {
            const progressHTML = `
                <div class="form-progress mt-3">
                    <div class="d-flex justify-content-between mb-2">
                        <small class="text-muted">Form Completion</small>
                        <small class="text-muted"><span id="progress-text">0%</span></small>
                    </div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar bg-primary" id="form-progress-bar" style="width: 0%"></div>
                    </div>
                </div>
            `;
            formHeader.insertAdjacentHTML('beforeend', progressHTML);
        }
    }

    function updateFormProgress() {
        const form = document.getElementById('predictionForm');
        if (!form) return;

        const requiredFields = form.querySelectorAll('[required]');
        const filledFields = Array.from(requiredFields).filter(field => {
            return field.value !== '' && field.checkValidity();
        });

        const progress = (filledFields.length / requiredFields.length) * 100;
        const progressBar = document.getElementById('form-progress-bar');
        const progressText = document.getElementById('progress-text');

        if (progressBar && progressText) {
            progressBar.style.width = progress + '%';
            progressText.textContent = Math.round(progress) + '%';
            
            if (progress === 100) {
                progressBar.classList.remove('bg-primary');
                progressBar.classList.add('bg-success');
            } else {
                progressBar.classList.remove('bg-success');
                progressBar.classList.add('bg-primary');
            }
        }
    }

    // Result page enhancements
    if (window.location.pathname.includes('result')) {
        initResultPage();
    }

    function initResultPage() {
        // Animate result appearance
        const resultCard = document.querySelector('.result-card');
        if (resultCard) {
            resultCard.style.opacity = '0';
            resultCard.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                resultCard.style.transition = 'all 0.8s ease';
                resultCard.style.opacity = '1';
                resultCard.style.transform = 'translateY(0)';
            }, 300);
        }

        // Animate progress bar
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            const targetWidth = progressBar.style.width;
            progressBar.style.width = '0%';
            
            setTimeout(() => {
                progressBar.style.transition = 'width 2s ease-in-out';
                progressBar.style.width = targetWidth;
            }, 800);
        }

        // Add result sharing functionality
        addResultSharing();
    }

    function addResultSharing() {
        // Add copy result functionality
        const actionButtons = document.querySelector('.action-buttons');
        if (actionButtons) {
            const copyBtn = document.createElement('button');
            copyBtn.className = 'btn btn-outline-secondary btn-lg ms-2';
            copyBtn.innerHTML = '<i class="fas fa-copy me-2"></i>Copy Results';
            copyBtn.onclick = copyResultsToClipboard;
            actionButtons.appendChild(copyBtn);
        }
    }

    function copyResultsToClipboard() {
        const resultText = document.querySelector('.result-card .card-body').innerText;
        navigator.clipboard.writeText(resultText).then(() => {
            showToast('Results copied to clipboard!', 'success');
        }).catch(() => {
            showToast('Failed to copy results', 'error');
        });
    }

    // Toast notification system
    function showToast(message, type = 'info') {
        const toastContainer = getOrCreateToastContainer();
        const toastId = 'toast-' + Date.now();
        
        const toastHTML = `
            <div class="toast align-items-center text-bg-${type} border-0" id="${toastId}" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-${type === 'success' ? 'check-circle' : (type === 'error' ? 'exclamation-circle' : 'info-circle')} me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        
        // Remove toast element after it's hidden
        toastElement.addEventListener('hidden.bs.toast', function() {
            this.remove();
        });
    }

    function getOrCreateToastContainer() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        return container;
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.getElementById('predictionForm');
            if (form && document.activeElement.tagName !== 'BUTTON') {
                form.requestSubmit();
            }
        }
    });

    // Initialize any additional animations
    initAnimations();

    function initAnimations() {
        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.card, .form-section').forEach(el => {
            observer.observe(el);
        });
    }
});

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    .animate-fade-in {
        animation: fadeIn 0.8s ease forwards;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .navbar-scrolled {
        background-color: rgba(0, 123, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }

    .form-progress {
        max-width: 300px;
        margin: 0 auto;
    }

    .toast-container {
        z-index: 9999;
    }
`;
document.head.appendChild(style);
