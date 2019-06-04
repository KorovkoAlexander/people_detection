
#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <math.h>

#include <gst/gst.h>
#include <gst/video/video.h>
#include <msgpack.h>
#include <opencv/cxcore.h>
#include <opencv/cv.h>

#include "gstdrawpredicts.h"

GST_DEBUG_CATEGORY_STATIC (gst_drawpredicts_debug);
#define GST_CAT_DEFAULT gst_drawpredicts_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_NO_LAG_TEXT
};

/* the capabilities of the inputs and outputs.
 *
 */

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format = { BGR }")
    );


static GstStaticPadTemplate video_sink_factory = GST_STATIC_PAD_TEMPLATE ("video_sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format = { BGR }")
    );


static GstStaticPadTemplate predicts_sink_factory = GST_STATIC_PAD_TEMPLATE ("predicts_sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("application/msgpack-predicts")
    );

static GstStaticPadTemplate meter_sink_factory = GST_STATIC_PAD_TEMPLATE ("meter_sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("application/meter")
    );

#define gst_drawpredicts_parent_class parent_class
G_DEFINE_TYPE (Gstdrawpredicts, gst_drawpredicts, GST_TYPE_ELEMENT);

static void gst_drawpredicts_finalize (GObject * object);
static void gst_drawpredicts_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_drawpredicts_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_drawpredicts_video_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);
static gboolean gst_drawpredicts_predicts_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);
static gboolean gst_drawpredicts_meter_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);
static GstFlowReturn gst_drawpredicts_video_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);
static GstFlowReturn gst_drawpredicts_predicts_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);
static GstFlowReturn gst_drawpredicts_meter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);


static GArray *
alloc_colors_pool(int len) {
  GArray * cvColorsPool = g_array_sized_new(
    TRUE, // zero_terminated
    FALSE, // clear_
    sizeof(CvScalar),
    len  // reserved_size
  );

  GRand * rnd = g_rand_new_with_seed(42);

  for (int i = 0; i < len; i++) {
    int b = g_rand_int_range(rnd, 0, 255);
    int g = g_rand_int_range(rnd, 0, 255);
    int r = g_rand_int_range(rnd, 0, 255);
    CvScalar color = cvScalar(b, g, r, 0);
    g_array_insert_val(cvColorsPool, i, color);
  }

  g_rand_free(rnd);

  return cvColorsPool;
}

/* GObject vmethod implementations */

/* initialize the drawpredicts's class */
static void
gst_drawpredicts_class_init (GstdrawpredictsClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_drawpredicts_set_property;
  gobject_class->get_property = gst_drawpredicts_get_property;
  gobject_class->finalize = gst_drawpredicts_finalize;

  g_object_class_install_property (gobject_class, PROP_NO_LAG_TEXT,
      g_param_spec_boolean ("no-lag-text", "No Lag Text", "Produce `lag` caption ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple(gstelement_class,
    "drawpredicts",
    "FIXME:Generic",
    "FIXME:Generic Template Element",
    "kostya <<user@hostname.org>>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&video_sink_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&predicts_sink_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&meter_sink_factory));
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_drawpredicts_init (Gstdrawpredicts * filter)
{
  filter->video_sinkpad = gst_pad_new_from_static_template (&video_sink_factory, "video_sink");
  gst_pad_set_event_function (filter->video_sinkpad,
                              GST_DEBUG_FUNCPTR(gst_drawpredicts_video_sink_event));
  gst_pad_set_chain_function (filter->video_sinkpad,
                              GST_DEBUG_FUNCPTR(gst_drawpredicts_video_chain));
  GST_PAD_SET_PROXY_CAPS (filter->video_sinkpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->video_sinkpad);


  filter->predicts_sinkpad = gst_pad_new_from_static_template (&predicts_sink_factory, "predicts_sink");
  gst_pad_set_event_function (filter->predicts_sinkpad,
                              GST_DEBUG_FUNCPTR(gst_drawpredicts_predicts_sink_event));
  gst_pad_set_chain_function (filter->predicts_sinkpad,
                              GST_DEBUG_FUNCPTR(gst_drawpredicts_predicts_chain));
  gst_element_add_pad (GST_ELEMENT (filter), filter->predicts_sinkpad);

  filter->meter_sinkpad = gst_pad_new_from_static_template (&meter_sink_factory, "meter_sink");
  gst_pad_set_event_function (filter->meter_sinkpad,
                              GST_DEBUG_FUNCPTR(gst_drawpredicts_meter_sink_event));
  gst_pad_set_chain_function (filter->meter_sinkpad,
                              GST_DEBUG_FUNCPTR(gst_drawpredicts_meter_chain));
  gst_element_add_pad (GST_ELEMENT (filter), filter->meter_sinkpad);


  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);

  filter->no_lag_text = FALSE;

  msgpack_zone_init(&filter->mempool, 2048);
  filter->cvColorsPool = alloc_colors_pool(1000);

  cvInitFont(
    &filter->bbox_font,
    CV_FONT_HERSHEY_SIMPLEX,
    0.8, // hscale
    0.8, // vscale
    0, // shear
    2, // thickness
    8  // line_type
  );
  cvInitFont(
    &filter->bbox_outer_font,
    CV_FONT_HERSHEY_SIMPLEX,
    0.8, // hscale
    0.8, // vscale
    0, // shear
    6, // thickness
    8  // line_type
  );
}

static void
gst_drawpredicts_finalize (GObject * object)
{
  Gstdrawpredicts *filter = GST_DRAWPREDICTS (object);

  msgpack_zone_destroy(&filter->mempool);
  g_array_unref(filter->cvColorsPool);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_drawpredicts_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstdrawpredicts *filter = GST_DRAWPREDICTS (object);

  switch (prop_id) {
    case PROP_NO_LAG_TEXT:
      filter->no_lag_text = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_drawpredicts_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstdrawpredicts *filter = GST_DRAWPREDICTS (object);

  switch (prop_id) {
    case PROP_NO_LAG_TEXT:
      g_value_set_boolean (value, filter->no_lag_text);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_drawpredicts_video_sink_setcaps (Gstdrawpredicts * self,
    GstCaps * caps)
{
  gboolean ret = TRUE;
  GstVideoInfo info;

  GST_DEBUG_OBJECT (self, "Setting caps: %" GST_PTR_FORMAT, caps);

  if (!gst_video_info_from_caps (&info, caps)) {
    GST_ERROR_OBJECT (self, "Failed to parse caps");
    ret = FALSE;
    goto out;
  }

  self->width = GST_VIDEO_INFO_WIDTH (&info);
  self->height = GST_VIDEO_INFO_HEIGHT (&info);

  // Ensure downstream uses the same caps:
  ret = gst_pad_set_caps (self->srcpad, caps);

out:
  return ret;
}

/* GstElement vmethod implementations */

/* this function handles sink events */
static gboolean
gst_drawpredicts_video_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  Gstdrawpredicts *filter;
  gboolean ret;

  filter = GST_DRAWPREDICTS (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps * caps;

      gst_event_parse_caps (event, &caps);
      ret = gst_drawpredicts_video_sink_setcaps (filter, caps);
      if (!ret) {
        goto done;
      }

      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }

done:
  return ret;
}

static gboolean
gst_drawpredicts_predicts_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  Gstdrawpredicts *filter;
  gboolean ret;

  filter = GST_DRAWPREDICTS (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps * caps;

      gst_event_parse_caps (event, &caps);
      ret = TRUE;
      goto done;
      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }

done:
  return ret;
}

static gboolean
gst_drawpredicts_meter_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  Gstdrawpredicts *filter;
  gboolean ret;

  filter = GST_DRAWPREDICTS (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps * caps;

      gst_event_parse_caps (event, &caps);
      ret = TRUE;
      goto done;
      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }

done:
  return ret;
}

void gst_drawpredicts_cv_put_text_with_border(Gstdrawpredicts *filter,
    CvArr* img, const char* text, CvPoint org)
{
  cvPutText(
    img, text,
    org,
    &filter->bbox_outer_font,
    cvScalar(0, 0, 0, 0)
  );
  cvPutText(
    img, text,
    org,
    &filter->bbox_font,
    cvScalar(255, 255, 255, 0)
  );
}

static gboolean
gst_drawpredicts_parse_predicts_payload(Gstdrawpredicts *filter,
    msgpack_object * deserialized, GstdrawpredictsParsedPredicts * parsed)
{

  // Payload:
  // [shape, _labels, _scores, _coords, _tracks]
  g_return_val_if_fail (deserialized->type == MSGPACK_OBJECT_ARRAY, FALSE);
  g_return_val_if_fail (deserialized->via.array.size == 5, FALSE);

  // 0: shape
  g_return_val_if_fail (deserialized->via.array.ptr[0].type == MSGPACK_OBJECT_ARRAY, FALSE);
  msgpack_object_array pred_shape = deserialized->via.array.ptr[0].via.array;
  g_return_val_if_fail (pred_shape.size == 2, FALSE);
  parsed->predict_width = pred_shape.ptr[0].via.u64;
  parsed->predict_height = pred_shape.ptr[1].via.u64;

  // 3: _coords
  g_return_val_if_fail (deserialized->via.array.ptr[3].type == MSGPACK_OBJECT_ARRAY, FALSE);
  parsed->coords = &deserialized->via.array.ptr[3].via.array;
  parsed->predicts_count = parsed->coords->size;

  // 1: _labels
  g_return_val_if_fail (deserialized->via.array.ptr[1].type == MSGPACK_OBJECT_ARRAY, FALSE);
  parsed->labels = &deserialized->via.array.ptr[1].via.array;
  g_return_val_if_fail (parsed->labels->size == parsed->predicts_count, FALSE);

  // 2: _scores
  if (deserialized->via.array.ptr[2].type == MSGPACK_OBJECT_NIL) {
    parsed->scores = NULL;
  } else {
    g_return_val_if_fail (deserialized->via.array.ptr[2].type == MSGPACK_OBJECT_ARRAY, FALSE);
    parsed->scores = &deserialized->via.array.ptr[2].via.array;
    g_return_val_if_fail (parsed->scores->size == parsed->predicts_count, FALSE);
  }

  // 4: _tracks
  if (deserialized->via.array.ptr[4].type == MSGPACK_OBJECT_NIL) {
    parsed->tracks = NULL;
  } else {
    g_return_val_if_fail (deserialized->via.array.ptr[4].type == MSGPACK_OBJECT_ARRAY, FALSE);
    parsed->tracks = &deserialized->via.array.ptr[4].via.array;
    g_return_val_if_fail (parsed->tracks->size == parsed->predicts_count, FALSE);
  }

  return TRUE;
}


static GstFlowReturn
gst_drawpredicts_maybe_render(Gstdrawpredicts *filter)
{

  GST_OBJECT_LOCK (filter);

  if (!filter->queued_video_buf || !filter->queued_predicts_buf) {
    goto fail_unlock;
  }

  GstClockTime diff_predicts = filter->queued_video_buf->pts - filter->queued_predicts_buf->pts;

  GstClockTime diff_meter = -1;
  if(filter->queued_meter_buf != NULL){
    diff_meter = filter->queued_video_buf->pts - filter->queued_meter_buf->pts;
  }

  msgpack_object deserialized;
  GstdrawpredictsParsedPredicts predicts;

  GstMapInfo iinfo;
  unsigned long magic;
  unsigned long expected_magic;
  // ############################################

  gst_buffer_map (filter->queued_predicts_buf, &iinfo, GST_MAP_READ);

  unsigned long len;
  guint8 *data = iinfo.data;
  magic = GST_READ_UINT32_BE(data);
  data += 4;
  len = GST_READ_UINT32_BE(data);
  data += 4;

  size_t header_len = 4 + 4;

  expected_magic = 0xfa91ffffUL;
  if (magic != expected_magic) {
    GST_ERROR ("unexpected predicts magic %lx", magic);
    gst_buffer_unmap (filter->queued_predicts_buf, &iinfo);
    goto fail_unlock;
  }

  if (header_len + len > iinfo.size) {
    GST_ERROR ("payload is larger than buffer (%lu > %lu)", header_len + len, iinfo.size);
    gst_buffer_unmap (filter->queued_predicts_buf, &iinfo);
    goto fail_unlock;
  }

  msgpack_unpack((const char *)data, len, NULL, &filter->mempool, &deserialized);
  data = NULL;

  gst_buffer_unmap (filter->queued_predicts_buf, &iinfo);

  // ############################################

  //  msgpack_object_print(stdout, deserialized);

  if (!gst_drawpredicts_parse_predicts_payload(filter, &deserialized, &predicts)) {
    GST_ERROR ("unable to parse the predicts payload");
    goto fail_unlock;
  }

  // ############################################
  unsigned long mean_count = -1, mean_count_kitchen = -1;

  if(filter->queued_meter_buf!= NULL)
  {
    gst_buffer_map (filter->queued_meter_buf, &iinfo, GST_MAP_READ);
    guint8 *data = iinfo.data;
    magic = GST_READ_UINT32_BE(data);
    data += 4;
    mean_count = GST_READ_UINT32_BE(data);
    data += 4;
    mean_count_kitchen = GST_READ_UINT32_BE(data);
    data += 4;

    expected_magic = 0xfa91f534UL;
    if (magic != expected_magic) {
      GST_ERROR ("unexpected meter magic %lx", magic);
      gst_buffer_unmap (filter->queued_meter_buf, &iinfo);
      goto fail_unlock;
    }

    gst_buffer_unmap (filter->queued_meter_buf, &iinfo);
  }

  // ############################################

  GstBuffer * outbuf = gst_buffer_copy(filter->queued_video_buf);

  GST_OBJECT_UNLOCK (filter);
  // Actually the access to `predicts` variable should be serialized as well,
  // but it looks like we don't need to hold the lock on `filter` for that,
  // because the `chain` func is serialized as well.

  gst_buffer_map (outbuf, &iinfo, GST_MAP_WRITE | GST_MAP_READ);

  IplImage * img = NULL;
  img = cvCreateImageHeader(cvSize(filter->width, filter->height), IPL_DEPTH_8U, 3);
  cvSetData(img, iinfo.data, CV_AUTOSTEP);


  char text[50];
  if (!filter->no_lag_text) {
    sprintf(text, "lag %ld ms", diff_predicts / 1000000);
    gst_drawpredicts_cv_put_text_with_border(
      filter, img, text, cvPoint(filter->width - 300, 50)
    );

    if (diff_meter != -1) {
      sprintf(text, "lag meter %ld ms", diff_meter / 1000000);
      gst_drawpredicts_cv_put_text_with_border(
        filter, img, text, cvPoint(filter->width - 300, 80)
      );
    }
  }

  if (mean_count != -1U) {
    sprintf(text, "People # %lu", mean_count);
    gst_drawpredicts_cv_put_text_with_border(
      filter, img, text, cvPoint(filter->width - 300, 110)
    );
  }

  if (mean_count_kitchen != -1U) {
    sprintf(text, "People in kitchen # %lu", mean_count_kitchen);
    gst_drawpredicts_cv_put_text_with_border(
      filter, img, text, cvPoint(filter->width - 320, 140)
    );
  }

//  GST_DEBUG("w %lu h %lu s %lu", pred_width, pred_height, predicts_count);

  double coeff_w = 1.0 * filter->width / predicts.predict_width;
  double coeff_h = 1.0 * filter->height / predicts.predict_height;

  for (int i = 0; i < predicts.predicts_count; i++) {
    unsigned long label = predicts.labels->ptr[i].via.u64;
    double score = -1;
    if (predicts.scores != NULL) {
      score = predicts.scores->ptr[i].via.f64;
    }
    double p0 = predicts.coords->ptr[i].via.array.ptr[0].via.f64;
    double p1 = predicts.coords->ptr[i].via.array.ptr[1].via.f64;
    double p2 = predicts.coords->ptr[i].via.array.ptr[2].via.f64;
    double p3 = predicts.coords->ptr[i].via.array.ptr[3].via.f64;

    CvScalar trackColor = cvScalar(255, 0, 0, 0);  // BGR
    if (predicts.tracks != NULL) {
      unsigned long track_idx = predicts.tracks->ptr[i].via.u64;
      trackColor = g_array_index(filter->cvColorsPool, CvScalar, track_idx % filter->cvColorsPool->len);
    }

    if (score > 0) {
      sprintf(text, "score=%.3f", score);
      gst_drawpredicts_cv_put_text_with_border(
        filter, img, text,
        cvPoint(p0 * coeff_w, p1 * coeff_h - 30)
      );
    }

    sprintf(text, "label=%lu", label);
    gst_drawpredicts_cv_put_text_with_border(
      filter, img, text,
      cvPoint(p0 * coeff_w, p1 * coeff_h - 10)
    );

    cvRectangle(
      img,
      cvPoint(p0 * coeff_w, p1 * coeff_h),
      cvPoint(p2 * coeff_w, p3 * coeff_h),
      trackColor,
      2, // thickness
      8, // lineType
      0  // shift
    );
  }

  cvReleaseImageHeader(&img);

  gst_buffer_unmap (outbuf, &iinfo);

  // ############################################

  // `gst_pad_push` should not be inside the lock, otherwise it deadlocks.
  return gst_pad_push (filter->srcpad, outbuf);

fail_unlock:
  GST_OBJECT_UNLOCK (filter);
  return GST_FLOW_OK;
}

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_drawpredicts_video_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Gstdrawpredicts *filter;
  GstFlowReturn ret;

  filter = GST_DRAWPREDICTS (parent);

  GST_OBJECT_LOCK (filter);
  if (filter->queued_video_buf)
    gst_buffer_unref (filter->queued_video_buf);
  filter->queued_video_buf = buf;
  GST_OBJECT_UNLOCK (filter);

  ret = gst_drawpredicts_maybe_render(filter);

  return ret;
}

static GstFlowReturn
gst_drawpredicts_predicts_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Gstdrawpredicts *filter;

  filter = GST_DRAWPREDICTS (parent);

  GST_OBJECT_LOCK (filter);
  if (filter->queued_predicts_buf)
    gst_buffer_unref (filter->queued_predicts_buf);
  filter->queued_predicts_buf = buf;
  GST_OBJECT_UNLOCK (filter);

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_drawpredicts_meter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Gstdrawpredicts *filter;

  filter = GST_DRAWPREDICTS (parent);

  GST_OBJECT_LOCK (filter);
  if (filter->queued_meter_buf)
    gst_buffer_unref (filter->queued_meter_buf);
  filter->queued_meter_buf = buf;
  GST_OBJECT_UNLOCK (filter);

  return GST_FLOW_OK;
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
drawpredicts_init (GstPlugin * drawpredicts)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template drawpredicts' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_drawpredicts_debug, "drawpredicts",
      0, "Template drawpredicts");

  return gst_element_register (drawpredicts, "drawpredicts", GST_RANK_NONE,
      GST_TYPE_DRAWPREDICTS);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "drawpredicts"
#endif

/* gstreamer looks for this structure to register drawpredictss
 *
 * exchange the string 'Template drawpredicts' with your drawpredicts description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    drawpredicts,
    "Template drawpredicts",
    drawpredicts_init,
    VERSION,
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)
