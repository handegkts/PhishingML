// Ana JavaScript dosyası

// Sayfadaki uyarı mesajlarını otomatik olarak kapat
document.addEventListener('DOMContentLoaded', function() {
    // Bootstrap uyarı mesajları için kapatma işlevi
    var alerts = document.querySelectorAll('.alert');
    
    alerts.forEach(function(alert) {
        setTimeout(function() {
            var dismiss = new bootstrap.Alert(alert);
            dismiss.close();
        }, 5000); // 5 saniye sonra kapat
    });
}); 