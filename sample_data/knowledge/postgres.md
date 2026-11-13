# PostgreSQL connectivity

When an application cannot connect to PostgreSQL, check the following:

1. Confirm that the PostgreSQL service is running.
2. Verify the database host and port.
3. Check that the database user has permission to connect.
4. Review firewall and network-policy rules.
5. Confirm that the application has the correct connection string.

Do not expose database credentials in application logs or source control.