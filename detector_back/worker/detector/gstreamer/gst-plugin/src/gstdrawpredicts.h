#ifndef __GST_DRAWPREDICTS_H__
#define __GST_DRAWPREDICTS_H__

#include <gst/gst.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_DRAWPREDICTS \
  (gst_drawpredicts_get_type())
#define GST_DRAWPREDICTS(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DRAWPREDICTS,Gstdrawpredicts))
#define GST_DRAWPREDICTS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DRAWPREDICTS,GstdrawpredictsClass))
#define GST_IS_DRAWPREDICTS(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DRAWPREDICTS))
#define GST_IS_DRAWPREDICTS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DRAWPREDICTS))

typedef struct _Gstdrawpredicts      Gstdrawpredicts;
typedef struct _GstdrawpredictsClass GstdrawpredictsClass;

struct _Gstdrawpredicts
{
  GstElement element;

  GstPad *video_sinkpad, *meter_sinkpad, *predicts_sinkpad, *srcpad;

  gint                     width;
  gint                     height;

  GstBuffer *queued_video_buf, *queued_meter_buf, *queued_predicts_buf;

  msgpack_zone             mempool;
  CvFont                   bbox_font;
  CvFont                   bbox_outer_font;
  GArray                   *cvColorsPool;

  gboolean no_lag_text;
};

struct _GstdrawpredictsClass
{
  GstElementClass parent_class;
};

GType gst_drawpredicts_get_type (void);

G_END_DECLS


typedef struct {
  unsigned long predict_width;
  unsigned long predict_height;
  unsigned long predicts_count;

  msgpack_object_array * labels;
  msgpack_object_array * scores;
  msgpack_object_array * coords;
  msgpack_object_array * tracks;

} GstdrawpredictsParsedPredicts;

#endif /* __GST_DRAWPREDICTS_H__ */
