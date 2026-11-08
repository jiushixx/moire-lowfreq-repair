# 上传GitHub

只上传本文件夹内容，不上传上一级outputs、个人文件、原图、缓存或大结果ZIP。

1. 在GitHub新建空仓库，名称可用moire-lowfreq-repair；可先设Private给老师访问。
2. 通过Git客户端推送整个项目（保留目录），或网页Add file → Upload files拖入目录内容。不要只上传一个ZIP，让老师能直接看README和代码。
3. Git方式，在本项目目录执行以下命令，将YOUR_NAME改成账号；这只是操作说明，本次整理未替你发布或创建远程仓库。

```bash
git init
git add .
git status
git commit -m "Organize moire restoration project and frozen evaluation"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/moire-lowfreq-repair.git
git push -u origin main
```

4. 检查README渲染、notebooks、results/frozen/summary.csv可见。
5. 权重包和DCID训练小子集单独交给老师，或在确认授权后通过适当附件渠道分享；README中明确如何获得。代码仓库不包含数据或权重，单下载仓库不能省略这些依赖。
6. 学校要求的姓名、学号、课程、作者分工自行填入README；本次没有猜写作者身份。未自动添加开源许可证，以免把第三方许可与自己代码许可混为一谈。
