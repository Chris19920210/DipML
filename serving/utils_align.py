import redis
import jieba
from nltk import word_tokenize as en_tokenize
import subprocess
import threading


def consume(s):
    for _ in s:
        pass


class AlignerClient(object):
    def __init__(self,
                 aligner_path,
                 fwd_param,
                 fwd_err,
                 rev_param,
                 rev_err,
                 dict_path,
                 redis_host,
                 redis_port,
                 timeout_secs):
        cmd = "/usr/bin/python2.7 " + aligner_path + " " + fwd_param + " " + fwd_err + " " + rev_param + " " + rev_err
        self.obj = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, shell=True)
        threading.Thread(target=consume, args=(self.obj.stderr,)).start()
        pool = redis.ConnectionPool(host=redis_host, port=redis_port, decode_responses=True)
        self.rcon = redis.Redis(connection_pool=pool)
        jieba.load_userdict(dict_path)
        self.zh_en = True

    def search_word_pair(self, org, trg):
        key = org + '__' + trg
        sname = 'fwd' if self.zh_en else 'rev'
        return self.rcon.hget(sname, key)

    def query(self, jobj):
        org_term = jobj['term']['origin']
        trg_term = jobj['term']['translate']
        res = []
        for req in jobj['data']:
            pid = req['pid']
            reqid = req['id']
            originl = req['origin'].replace("\n", "")
            translate = req['translate'].replace("\n", "")
            if self.zh_en:
                origin = ' '.join(jieba.cut(originl.strip()))
                translate = ' '.join(en_tokenize(translate.strip()))
            else:
                translate = ' '.join(jieba.cut(translate.strip()))
                origin = ' '.join(en_tokenize(originl.strip()))
            line = origin + ' ||| ' + translate
            '''if org_term in originl:
                template = r""
                for l in org_term:
                    template += l + "(\s*)"
                print('template...>', template)
                pterm = re.compile(template)
                line = pterm.sub(org_term + ' ', line)'''
            orgs = origin.split(" ")
            trgs = translate.split(" ")
            self.obj.stdin.write('{}\n'.format(line.strip()))
            self.obj.stdin.flush()
            aline = self.obj.stdout.readline()
            pairs = aline.split(" ")
            candidates, zh_candidates = [], []
            for p in pairs:
                cols = p.split("-")
                if len(cols) != 2:
                    continue
                trg = trgs[int(cols[1])]
                pair_log = self.search_word_pair(orgs[int(cols[0])], trg)
                pair_log = -1000.0 if pair_log is None else pair_log
                if orgs[int(cols[0])] == org_term:
                    candidates.append([int(cols[1]), trg, float(pair_log)])
            if len(candidates) > 0:
                start, end, maxid, maxlog = -1, -1, -1, -1000
                phrase = False
                cnt = 0
                for idx, trg, log in candidates:
                    if log > -2.0:
                        if start == -1:
                            start = idx
                            end = idx + 1
                        elif cnt == len(candidates) - 1 and idx == end:
                            phrase = True
                            for k in range(start, end + 1):
                                if k == start:
                                    trgs[k] = trg_term
                                else:
                                    trgs[k] = ''
                        else:
                            if idx == end:
                                end = idx + 1
                            elif end - start > 1:
                                phrase = True
                                for k in range(start, end):
                                    if k == start:
                                        trgs[k] = trg_term
                                    else:
                                        trgs[k] = ''
                                start = idx
                                end = idx + 1
                            else:
                                start = idx
                                end = idx + 1
                    else:
                        if start != -1 and end - start > 1:
                            phrase = True
                            for k in range(start, end):
                                if k == start:
                                    trgs[k] = trg_term
                                else:
                                    trgs[k] = ''
                        if start != -1:
                            start = -1
                            end = -1
                    cnt += 1
                if not phrase:
                    for idx, trg, log in candidates:
                        if log > maxlog:
                            maxid = idx
                            maxlog = log
                    trgs[maxid] = trg_term
            print(candidates)
            ndic = {"pid": pid, "id": reqid, "origin": originl, "translate": ' '.join(trgs)}
            res.append(ndic)
        return {"status": 0, "msg": "", "data": res}
