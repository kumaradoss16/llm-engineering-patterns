# Restarting nginx

To restart nginx on a Linux system, use:

systemctl restart nginx

To reload configuration without stopping the service, use:

systemctl reload nginx

Before reloading, validate the configuration with:

nginx -t

If nginx fails to start, inspect the service logs and check whether another
process is already using the configured port.