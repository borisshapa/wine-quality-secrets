FROM mcr.microsoft.com/mssql/server

ARG MSSQL_UID="sa"
ARG MSSQL_PWD="Droider0ru208504AU"

ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=${MSSQL_PWD}
ENV USER_ID=${MSSQL_UID}

USER root

CMD ./opt/mssql/bin/sqlservr