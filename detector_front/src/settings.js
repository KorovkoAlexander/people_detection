const protocol = window.location.protocol;
const hostname = window.location.hostname;
const port = window.location.port;

const hostport = hostname + (port === "80"? "" : ":" + port)

export const apiRoot = '${protocol}//${host}';