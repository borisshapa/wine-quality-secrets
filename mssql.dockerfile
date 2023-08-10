FROM mcr.microsoft.com/mssql/server

USER root

ARG PASSWORD="Droider0ru208504AU"
ARG USER_ID="sa"

ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=${PASSWORD}
ENV USER_ID=${USER_ID}

COPY db/ /

RUN chmod +x /import_data.sh

CMD ./import_data.sh & /opt/mssql/bin/sqlservr