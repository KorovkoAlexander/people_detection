import argparse
import asyncio
import os
import ipaddress
import json
import logging
import sys

import netifaces as ni
import websockets

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst, GstSdp, GstWebRTC


Gst.init(None)

PIPELINE_DESC = """
 tcpclientsrc host=127.0.0.1 port=33141 do-timestamp=true
 ! application/x-rtp-stream,encoding-name=VP8
 ! rtpstreamdepay
 ! rtpjitterbuffer
 ! rtpvp8depay
 ! rtpvp8pay
 ! application/x-rtp,media=video,encoding-name=VP8,payload=97
 ! webrtcbin name=sendrecv bundle-policy=max-bundle
"""

# Doesn't work for me in both Chrome and Firefox:
# ! x264enc tune=zerolatency speed-preset=superfast bitrate=1000
# ! rtph264pay
# ! application/x-rtp,media=video,encoding-name=H264,payload=96

logger = logging.getLogger("signalling")

logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ICE_ALLOWED_NETWORKS = []
KEEPALIVE_TIMEOUT = 0


def get_nic_networks(nic):
    networks = []
    ifaddresses = ni.ifaddresses(nic)
    for family, addresses in ifaddresses.items():
        for address in addresses:
            try:
                net = ipaddress.ip_network(
                    f'{address["addr"]}/{address["netmask"]}',
                    strict=False,  # ignore host bits
                )
            except (KeyError, ValueError):
                logger.info(
                    "Failed to parse a network on %s: %r", nic, address, exc_info=True
                )
            else:
                networks.append(net)
    assert networks
    return networks


async def recv_msg_ping(ws, raddr):
    """
    Wait for a message forever, and send a regular ping to prevent bad routers
    from closing the connection.
    """
    msg = None
    while msg is None:
        try:
            msg = await asyncio.wait_for(ws.recv(), KEEPALIVE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info("Sending keepalive ping to %r in recv", raddr)
            await ws.ping()
    return msg


def start_pipeline(conn):
    def send_ice_candidate_message(_, mlineindex, candidate):
        remote_ip, _ = conn.remote_address  # (host, port)
        # "candidate:26 2 TCP 1015023870 10.60.130.20 9 typ host tcptype active"
        candidate_ip = candidate.split(" ")[4]
        if not keep_ice_candidate(candidate_ip, remote_ip):
            logger.info("skipping candidate %r", candidate)
            return

        icemsg = json.dumps(
            {"ice": {"candidate": candidate, "sdpMLineIndex": mlineindex}}
        )
        logger.info("sending ice candidate %r", icemsg)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(conn.send(icemsg))

    def keep_ice_candidate(candidate_ip, ws_ip):
        # Filter out docker, localhost and other private ip addresses

        if not ICE_ALLOWED_NETWORKS:
            # If no nics were provided -- don't filter anything out.
            return True

        ip = ipaddress.ip_address(candidate_ip)
        for net in ICE_ALLOWED_NETWORKS:
            if ip in net:
                return True
        return False

    def on_negotiation_needed(element):
        promise = Gst.Promise.new_with_change_func(on_offer_created, element, None)
        element.emit("create-offer", None, promise)

    def send_sdp_offer(offer):
        text = offer.sdp.as_text()
        logger.info("Sending offer:\n%s", text)
        msg = json.dumps({"sdp": {"type": "offer", "sdp": text}})
        loop = asyncio.new_event_loop()
        loop.run_until_complete(conn.send(msg))

    def on_offer_created(promise, _, __):
        nonlocal webrtc
        promise.wait()
        reply = promise.get_reply()
        offer = reply["offer"]
        promise = Gst.Promise.new()
        webrtc.emit("set-local-description", offer, promise)
        promise.interrupt()
        send_sdp_offer(offer)

    pipe = Gst.parse_launch(PIPELINE_DESC)
    webrtc = pipe.get_by_name("sendrecv")
    webrtc.connect("on-negotiation-needed", on_negotiation_needed)
    webrtc.connect("on-ice-candidate", send_ice_candidate_message)
    pipe.set_state(Gst.State.PLAYING)
    return pipe, webrtc


async def connection_handler(ws):
    raddr = ws.remote_address
    try:
        pipe, webrtc = start_pipeline(ws)
    except Exception:
        ws.close()
        raise

    try:
        while True:
            # Receive command, wait forever if necessary
            msg = await recv_msg_ping(ws, raddr)
            handle_sdp(webrtc, msg)
    finally:
        ws.close()
        pipe.set_state(Gst.State.NULL)


def handle_sdp(webrtc, message):
    msg = json.loads(message)
    if "sdp" in msg:
        sdp = msg["sdp"]
        assert sdp["type"] == "answer"
        sdp = sdp["sdp"]
        logger.info("Received answer:\n%s", sdp)
        res, sdpmsg = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(bytes(sdp.encode()), sdpmsg)
        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
        )
        promise = Gst.Promise.new()
        webrtc.emit("set-remote-description", answer, promise)
        promise.interrupt()
    elif "ice" in msg:
        ice = msg["ice"]
        candidate = ice["candidate"]
        sdpmlineindex = ice["sdpMLineIndex"]
        webrtc.emit("add-ice-candidate", sdpmlineindex, candidate)


async def hello_peer(ws):
    """
    Exchange hello, register peer
    """
    raddr = ws.remote_address
    hello = await ws.recv()
    if hello != "HELLO":
        await ws.close(code=1002, reason="invalid protocol")
        raise Exception("Invalid hello from {!r}".format(raddr))
    # Send back a HELLO
    await ws.send("HELLO")


async def handler(ws, path):
    """
    All incoming messages are handled here. @path is unused.
    """
    raddr = ws.remote_address
    logger.info("Connected to %r", raddr)
    await hello_peer(ws)
    try:
        await connection_handler(ws)
    finally:
        logger.info("Connection to peer %r closed, exiting handler", raddr)


def main():
    global ICE_ALLOWED_NETWORKS, KEEPALIVE_TIMEOUT

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--addr", default="0.0.0.0", help="Address to listen on")
    parser.add_argument("--port", default=8443, type=int, help="Port to listen on")
    parser.add_argument(
        "--keepalive-timeout",
        dest="keepalive_timeout",
        default=30,
        type=int,
        help="Timeout for keepalive (in seconds)",
    )

    options = parser.parse_args(sys.argv[1:])

    addr_port = (options.addr, options.port)
    KEEPALIVE_TIMEOUT = options.keepalive_timeout

    # The list of IP networks which are allowed to be sent out as the ICE candidates.
    # This allows to filter out the internal and definitely unreachable interfaces,
    # such as docker bridges, loopback and so on.
    ICE_ALLOWED_NETWORKS = [
        network
        for nic in os.getenv("WS_SIG_INTERFACES_FOR_ICE").split(" ")
        for network in get_nic_networks(nic)
    ]

    logger.info("ICE allowed networks: %s", ICE_ALLOWED_NETWORKS)

    logger.info("Listening on https://{}:{}".format(*addr_port))
    # Websocket server
    wsd = websockets.serve(
        handler,
        *addr_port,
        # Maximum number of messages that websockets will pop
        # off the asyncio and OS buffers per connection. See:
        # https://websockets.readthedocs.io/en/stable/api.html#websockets.protocol.WebSocketCommonProtocol
        max_queue=16,
    )

    asyncio.get_event_loop().run_until_complete(wsd)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
