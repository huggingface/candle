def remove_prefix(text, prefix):
  return text[text.startswith(prefix) and len(prefix):]
nps = {}
for k, v in model.state_dict().items():
  k = remove_prefix(k, 'module_list.')
  nps[k] = v.detach().numpy()
np.savez('yolo-v3.ot', **nps)
