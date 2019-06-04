IMAGE    := rambler-ai/sota-gstreamer:$(shell id -nu)
OUTDIR   := "${HOME}"/out

all: build run

build:
	docker build --rm -t ${IMAGE} -f Dockerfile ..


# Stream pictures from all cameras
stream-save-photos:
	docker run -it --rm --net=host --runtime=nvidia -v "${OUTDIR}":/out ${IMAGE} \
		sh -c ' \
	GST_DEBUG=3 \
	gst-launch-1.0 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.59/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videorate ! video/x-raw,framerate=1/30 \
		! pngenc \
		! multifilesink location=/out/cam_59_%d.png \
	\
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.60/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videorate ! video/x-raw,framerate=1/30 \
		! pngenc \
		! multifilesink location=/out/cam_60_%d.png \
	\
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.61/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videorate ! video/x-raw,framerate=1/30 \
		! pngenc \
		! multifilesink location=/out/cam_61_%d.png \
	\
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.62/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videorate ! video/x-raw,framerate=1/30 \
		! pngenc \
		! multifilesink location=/out/cam_62_%d.png \
	\
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.83/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videorate ! video/x-raw,framerate=1/30 \
		! pngenc \
		! multifilesink location=/out/cam_83_%d.png \
	\
	'


# Stream a single video with a grid of all cameras
stream-all-cameras:
	docker run -it --rm --net=host --runtime=nvidia ${IMAGE} \
		sh -c ' \
	GST_DEBUG=3 \
	gst-launch-1.0 \
		videomixer name=mix \
			sink_0::xpos=0    sink_0::ypos=0 \
			sink_1::xpos=0    sink_1::ypos=0 \
			sink_2::xpos=640  sink_2::ypos=0 \
			sink_3::xpos=1280 sink_3::ypos=0 \
			sink_4::xpos=0    sink_4::ypos=480 \
			sink_5::xpos=640  sink_5::ypos=480 \
			sink_6::xpos=1280 sink_6::ypos=480 \
		! "video/x-raw,width=1920,height=960" \
		! videoconvert \
		! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
		! rtph264pay ! rtpstreampay ! tcpserversink host=0.0.0.0 port=34241 \
	\
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.59/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! timeoverlay ! textoverlay text=".59" ! mix.sink_1 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.60/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! timeoverlay ! textoverlay text=".60" ! mix.sink_2 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.61/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! timeoverlay ! textoverlay text=".61" ! mix.sink_3 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.62/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! timeoverlay ! textoverlay text=".62" ! mix.sink_4 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.83/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! timeoverlay ! textoverlay text=".83" ! mix.sink_5 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.95/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! timeoverlay ! textoverlay text=".95" ! mix.sink_6 \
	\
	'


################################################
#
#
# Step 1 (executed on gtx00): run the inference pipeline
demo-tcp-inf:
	docker run -it --rm --net=host --runtime=nvidia ${IMAGE} \
		sh -c ' \
		GST_DEBUG=3,drawpredicts:4,basetransform:3,python:4,fpsdisplaysink:5 GST_PLUGIN_PATH=$$GST_PLUGIN_PATH:$$PWD \
		gst-launch-1.0 \
			rtspsrc "location=rtsp://${CAMERAS_URL}.60/axis-media/media.amp?videocodec=h264&audio=0&compression=20&resolution=1280x960&fps=0" \
			! application/x-rtp \
			! rtpjitterbuffer \
			! rtph264depay \
			! decodebin ! videoconvert \
			! video/x-raw,width=1280,height=960 \
			! clockoverlay text=remote \
			! tee name=t \
			\
			t. ! dp.video_sink \
			\
			t. \
			! queue \
			! videorate ! video/x-raw,framerate=5/1 \
			! videoscale \
			! predictor_py \
			! tracker_py \
			! dp.predicts_sink \
			\
			drawpredicts name=dp \
			! videoconvert \
\
! mix.sink_1   videomixer name=mix sink_0::xpos=0    sink_0::ypos=0 \
\
			! x264enc tune=zerolatency speed-preset=superfast bitrate=2000 \
			! rtph264pay ! rtpstreampay ! tcpserversink host=0.0.0.0 port=34241 \
		'

# Step 2 (executed on a laptop): receive the resulting video stream
demo-tcp-recv:
	GST_DEBUG=3 \
	gst-launch-1.0 tcpclientsrc host=gtx00.ad.rambler.tech port=34241 do-timestamp=true \
		! application/x-rtp-stream \
		! rtpstreamdepay \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay ! decodebin ! videoconvert \
		! clockoverlay ypad=70 text=" local" \
		! autovideosink

# Alternative step 1 (executed on gtx00): stream a single video with a grid of all cameras with predicts
demo-tcp-inf-all:
	docker run -it --rm --net=host --runtime=nvidia ${IMAGE} \
		sh -c ' \
	GST_DEBUG=3,drawpredicts:4,basetransform:3,python:4,fpsdisplaysink:5 GST_PLUGIN_PATH=$$GST_PLUGIN_PATH:$$PWD \
	gst-launch-1.0 \
		videomixer name=mix \
			sink_0::xpos=0    sink_0::ypos=0 \
			sink_1::xpos=0    sink_1::ypos=0 \
			sink_2::xpos=640  sink_2::ypos=0 \
			sink_3::xpos=1280 sink_3::ypos=0 \
			sink_4::xpos=0    sink_4::ypos=480 \
			sink_5::xpos=640  sink_5::ypos=480 \
			sink_6::xpos=1280 sink_6::ypos=480 \
		! "video/x-raw,width=1920,height=960" \
		! videoconvert \
		! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
		! rtph264pay ! rtpstreampay ! tcpserversink host=0.0.0.0 port=34241 \
	\
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.59/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! tee name=t59 \
		t59. ! dp59.video_sink \
		t59. \
		! queue \
		! videorate ! video/x-raw,framerate=3/1 \
		! videoscale \
		! predictor_py gpu-device-id=1 \
		! tee name=p59 \
		p59. \
		! tracker_py min-hits=2 max-age=3 \
		! dp59.predicts_sink \
		p59. \
		! average_meter_py \
		! dp59.meter_sink \
		drawpredicts name=dp59 \
		! textoverlay text=".59" ! mix.sink_1 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.60/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! tee name=t60 \
		t60. ! dp60.video_sink \
		t60. \
		! queue \
		! videorate ! video/x-raw,framerate=3/1 \
		! videoscale \
		! predictor_py gpu-device-id=2 \
		! tee name=p60 \
		p60. \
		! tracker_py min-hits=2 max-age=3 \
		! dp60.predicts_sink \
		p60. \
		! average_meter_py \
		! dp60.meter_sink \
		drawpredicts name=dp60 \
		! textoverlay text=".60" ! mix.sink_2 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.61/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! tee name=t61 \
		t61. ! dp61.video_sink \
		t61. \
		! queue \
		! videorate ! video/x-raw,framerate=3/1 \
		! videoscale \
		! predictor_py gpu-device-id=3 \
		! tee name=p61 \
		p61. \
		! tracker_py min-hits=2 max-age=3 \
		! dp61.predicts_sink \
		p61. \
		! average_meter_py \
		! dp61.meter_sink \
		drawpredicts name=dp61 \
		! textoverlay text=".61" ! mix.sink_3 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.62/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! tee name=t62 \
		t62. ! dp62.video_sink \
		t62. \
		! queue \
		! videorate ! video/x-raw,framerate=3/1 \
		! videoscale \
		! predictor_py gpu-device-id=1 \
		! tee name=p62 \
		p62. \
		! tracker_py min-hits=2 max-age=3 \
		! dp62.predicts_sink \
		p62. \
		! average_meter_py \
		! dp62.meter_sink \
		drawpredicts name=dp62 \
		! textoverlay text=".62" ! mix.sink_4 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.83/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! tee name=t83 \
		t83. ! dp83.video_sink \
		t83. \
		! queue \
		! videorate ! video/x-raw,framerate=3/1 \
		! videoscale \
		! predictor_py gpu-device-id=2 \
		! tracker_py min-hits=2 max-age=3 \
		! tee name=p83 \
		p83. \
		! people_counter_py \
		! dp83.meter_sink \
		p83. \
		! dp83.predicts_sink \
		drawpredicts name=dp83 \
		! textoverlay text=".83" ! mix.sink_5 \
	\
		rtspsrc "location=rtsp://${CAMERAS_URL}.95/axis-media/media.amp?videocodec=h264&audio=0&compression=50&resolution=1280x960&fps=0" \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay \
		! decodebin ! videoconvert \
		! videoscale \
		! "video/x-raw,width=640,height=480" \
		! tee name=t95 \
		t95. ! dp95.video_sink \
		t95. \
		! queue \
		! videorate ! video/x-raw,framerate=3/1 \
		! videoscale \
		! predictor_py gpu-device-id=3 \
		! tee name=p95 \
		p95. \
		! tracker_py min-hits=2 max-age=3 \
		! dp95.predicts_sink \
		p95. \
		! average_meter_py \
		! dp95.meter_sink \
		drawpredicts name=dp95 \
		! textoverlay text=".95" ! mix.sink_6 \
	\
	'

# Alternative step 1 (executed on gtx00): accept a video stream and return it back (w/o inference)
demo-tcp-pass:
	docker run -it --rm --net=host --runtime=nvidia ${IMAGE} \
		sh -c ' \
		GST_DEBUG=3,python:4,fpsdisplaysink:5 GST_PLUGIN_PATH=$$GST_PLUGIN_PATH:$$PWD \
		gst-launch-1.0 \
			tcpserversrc host=127.0.0.1 port=14242 do-timestamp=true \
			! application/x-rtp-stream \
			! rtpstreamdepay \
			! application/x-rtp \
			! rtpjitterbuffer \
			! rtph264depay \
			! decodebin \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! rtph264pay ! rtpstreampay ! tcpserversink host=0.0.0.0 port=34241 \
		'

# Alternative step 1(executed on gtx00): accept a video stream, make copies and measure fps
demo-tcp-fps:
	docker run -it --rm --net=host --runtime=nvidia ${IMAGE} \
		sh -c ' \
		GST_DEBUG=3,python:4,fpsdisplaysink:5 GST_PLUGIN_PATH=$$GST_PLUGIN_PATH:$$PWD \
		gst-launch-1.0 \
			tcpserversrc host=0.0.0.0 port=34241 do-timestamp=true \
			! application/x-rtp-stream \
			! rtpstreamdepay \
			! application/x-rtp \
			! rtpjitterbuffer \
			! rtph264depay \
			! decodebin \
			! videorate ! video/x-raw,framerate=300/1 \
			! tee name=t \
			\
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! funnel name=f \
			! fpsdisplaysink text-overlay=false video-sink=fakesink \
			\
			t.src_1 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_1 \
			\
			t.src_2 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_2 \
			\
			t.src_3 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_3 \
			\
			t.src_4 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_4 \
			\
			t.src_5 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_5 \
			\
			t.src_6 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_6 \
			\
			t.src_7 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_7 \
			\
			t.src_8 \
			! queue \
			! videoscale \
			! video/x-raw,width=800,height=800 \
			! videoscale \
			! video/x-raw,width=1280,height=720 \
			! x264enc tune=zerolatency speed-preset=superfast bitrate=1000 \
			! f.sink_8 \
		'


################################################
#
# Test transmission of a stream locally via tcp
#
local-tcp-send:
	GST_DEBUG=3 \
	gst-launch-1.0 v4l2src ! jpegdec \
	! video/x-raw,width=1280,height=720 \
	! videorate ! video/x-raw,framerate=30/1 \
	! x264enc tune=zerolatency speed-preset=superfast bitrate=500 \
	! rtph264pay ! rtpstreampay ! tcpserversink host=127.0.0.1 port=14244

local-tcp-recv:
	GST_DEBUG=3 \
	gst-launch-1.0 tcpclientsrc host=gtx00.ad.rambler.tech port=34241 do-timestamp=true \
		! application/x-rtp-stream \
		! rtpstreamdepay \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay ! decodebin ! videoconvert \
		! autovideosink


################################################
#
# Test transmission of a stream locally via udp
#
local-udp-send:
	GST_DEBUG=3 \
	gst-launch-1.0 v4l2src ! jpegdec \
	! video/x-raw,width=1280,height=720 \
	! videorate ! video/x-raw,framerate=30/1 \
	! x264enc tune=zerolatency speed-preset=superfast bitrate=2000 \
	! rtph264pay ! udpsink host=gtx00.ad.rambler.tech port=34241

local-udp-recv:
	GST_DEBUG=3 \
	gst-launch-1.0 udpsrc port=34241 \
		! application/x-rtp \
		! rtpjitterbuffer \
		! rtph264depay ! decodebin ! videoconvert \
		! autovideosink


################################################
#
# Just read the webcam locally
#
local-vid:
	GST_DEBUG=3 \
	gst-launch-1.0 \
		v4l2src device=/dev/video0 ! jpegdec ! autovideosink
